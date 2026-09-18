"""
Local server for Mini TTS Web App.
Serves:
- Web App UI & static assets from web/
- Dataset audio wav files from km_kh_male/wavs/ at /audio/<id>.wav
- Dataset samples API at /api/samples
- Dataset stats API at /api/stats
- ONNX models and JSON dictionaries from web/models/
"""
import http.server
import json
import os
import re
import socketserver
import urllib.parse
import numpy as np
import onnxruntime as ort

PORT = 8002
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
DATASET_DIR = os.path.join(BASE_DIR, "km_kh_male")
WAVS_DIR = os.path.join(DATASET_DIR, "wavs")
SAMPLES_JSON = os.path.join(WEB_DIR, "data", "samples.json")
VOCAB_JSON = os.path.join(WEB_DIR, "models", "vocab.json")
LEXICON_JSON = os.path.join(WEB_DIR, "models", "lexicon.json")


_pipeline = None
_acoustic_ort = None
_vocoder_ort = None


class TTSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        # Enable CORS and SharedArrayBuffer headers needed for multi-threaded WASM / WebGPU
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/synthesize":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8")
            try:
                payload = json.loads(body) if body else {}
                text = payload.get("text", "")
                speed = float(payload.get("speed", 1.0))

                global _pipeline
                if _pipeline is None:
                    from infer import KhmerTTSPipeline
                    _pipeline = KhmerTTSPipeline()

                wav = _pipeline.synthesize(text, speed=speed)

                import io
                import soundfile as sf
                bio = io.BytesIO()
                sf.write(bio, wav, 22050, format="WAV")
                wav_bytes = bio.getvalue()

                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav_bytes)))
                self.end_headers()
                self.wfile.write(wav_bytes)
            except Exception as e:
                self.send_json_response({"error": str(e)}, status=500)
            return

        self.send_error(404, "Endpoint not found")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. API: Dataset Stats
        if path == "/api/stats":
            self.handle_api_stats()
            return

        # 2. API: Dataset Samples (with search & pagination)
        if path == "/api/samples":
            self.handle_api_samples(query)
            return

        # 3. Audio Streaming: /audio/<filename>
        if path.startswith("/audio/"):
            wav_name = os.path.basename(path)
            wav_path = os.path.join(WAVS_DIR, wav_name)
            if not wav_name.endswith(".wav"):
                wav_path += ".wav"

            if os.path.exists(wav_path):
                self.serve_audio_file(wav_path)
            else:
                self.send_error(404, f"Audio file not found: {wav_name}")
            return

        # Default: Serve web/ directory
        super().do_GET()

    def handle_api_stats(self):
        stats = {
            "dataset_name": "Khmer Male Speech (km_kh_male)",
            "total_wavs": len(os.listdir(WAVS_DIR)) if os.path.exists(WAVS_DIR) else 0,
            "sample_rate": 48000,
            "target_rate": 22050,
            "language": "Khmer (km)",
            "models": {
                "acoustic_onnx": os.path.exists(os.path.join(WEB_DIR, "models", "acoustic.onnx")),
                "vocoder_onnx": os.path.exists(os.path.join(WEB_DIR, "models", "vocoder.onnx")),
                "vocab_json": os.path.exists(VOCAB_JSON),
                "lexicon_json": os.path.exists(LEXICON_JSON),
            }
        }
        if os.path.exists(VOCAB_JSON):
            with open(VOCAB_JSON, encoding="utf-8") as f:
                stats["vocab_size"] = len(json.load(f))
        if os.path.exists(LEXICON_JSON):
            with open(LEXICON_JSON, encoding="utf-8") as f:
                stats["lexicon_words"] = len(json.load(f))

        self.send_json_response(stats)

    def handle_api_samples(self, query):
        q = query.get("q", [""])[0].strip().lower()
        page = int(query.get("page", ["1"])[0])
        limit = int(query.get("limit", ["20"])[0])

        if not os.path.exists(SAMPLES_JSON):
            self.send_json_response({"samples": [], "total": 0, "page": page, "limit": limit})
            return

        with open(SAMPLES_JSON, encoding="utf-8") as f:
            all_samples = json.load(f)

        if q:
            filtered = [s for s in all_samples if q in s["text"].lower() or q in s["id"].lower()]
        else:
            filtered = all_samples

        start = (page - 1) * limit
        end = start + limit
        paged = filtered[start:end]

        self.send_json_response({
            "samples": paged,
            "total": len(filtered),
            "page": page,
            "limit": limit
        })

    def serve_audio_file(self, file_path):
        try:
            file_size = os.path.getsize(file_path)
            range_header = self.headers.get("Range")

            if range_header:
                # Handle Byte-Range requests for seamless audio seeking
                range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
                if range_match:
                    start = int(range_match.group(1))
                    end = int(range_match.group(2)) if range_match.group(2) else file_size - 1
                    end = min(end, file_size - 1)
                    content_length = end - start + 1

                    self.send_response(206)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(content_length))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()

                    with open(file_path, "rb") as f:
                        f.seek(start)
                        self.wfile.write(f.read(content_length))
                    return

            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        except Exception as e:
            self.send_error(500, f"Error reading audio: {e}")

    def send_json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server():
    server_address = ("", PORT)
    with socketserver.TCPServer(server_address, TTSRequestHandler) as httpd:
        print(f"============================================================")
        print(f" Mini TTS Web App Server running at: http://localhost:{PORT}")
        print(f" Dataset Connected: {DATASET_DIR}")
        print(f" Audio Endpoint: http://localhost:{PORT}/audio/<utt_id>.wav")
        print(f"============================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    run_server()
