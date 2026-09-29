"""
Local Studio & Dataset Server for KHM-TtS (Khmer Text-to-Speech).
Serves:
- Web App UI & static assets from web/ (index.html, style.css, app.js)
- Dataset audio wav files from km_kh_male/wavs/ or processed/wavs_22k/ at /audio/<id>.wav
- Dataset samples API at /api/samples (with live search & pagination)
- Dataset statistics API at /api/stats
- Live Tokenizer Breakdown API at /api/tokenize
- Audio Synthesis API at /api/synthesize
- ONNX models and JSON dictionaries from web/models/
"""
import http.server
import json
import os
import re
import socketserver
import urllib.parse
import urllib.request

PORT = 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
DATASET_DIR = os.path.join(BASE_DIR, "km_kh_male")
WAVS_DIR = os.path.join(DATASET_DIR, "wavs")
PROCESSED_WAVS_DIR = os.path.join(BASE_DIR, "processed", "wavs_22k")
SAMPLES_JSON = os.path.join(WEB_DIR, "data", "samples.json")
VOCAB_JSON = os.path.join(WEB_DIR, "models", "vocab.json")
LEXICON_JSON = os.path.join(WEB_DIR, "models", "lexicon.json")

_pipeline = None
_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from data.khmer_tokenizer import KhmerTokenizer
        _tokenizer = KhmerTokenizer(vocab_path=VOCAB_JSON)
    return _tokenizer


class TTSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        # Enable CORS and SharedArrayBuffer headers needed for WebGPU / multi-threaded WASM
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

        # 1. API: Synthesize text -> WAV
        if parsed.path == "/api/synthesize":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
            try:
                payload = json.loads(body) if body else {}
                text = payload.get("text", "").strip()
                speed = float(payload.get("speed", 1.0))

                if not text:
                    self.send_json_response({"error": "Empty text provided"}, status=400)
                    return

                global _pipeline
                if _pipeline is None:
                    from infer import KhmerTTSPipeline
                    _pipeline = KhmerTTSPipeline(vocab_path=VOCAB_JSON)

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

        # 2. API: Tokenize text
        if parsed.path == "/api/tokenize":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
            try:
                payload = json.loads(body) if body else {}
                text = payload.get("text", "").strip()
                tok = get_tokenizer()
                normalized = tok.normalizer.normalize(text)
                tokens = tok.tokenize(text)
                ids = tok.text_to_ids(text)
                self.send_json_response({
                    "original": text,
                    "normalized": normalized,
                    "tokens": tokens,
                    "token_ids": ids,
                    "length": len(tokens)
                })
            except Exception as e:
                self.send_json_response({"error": str(e)}, status=500)
            return

        # 3. API: Translate text to Khmer
        if parsed.path == "/api/translate":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
            try:
                payload = json.loads(body) if body else {}
                text = payload.get("text", "").strip()
                if not text:
                    self.send_json_response({"error": "Empty text provided"}, status=400)
                    return

                target_lang = payload.get("target", "km")
                source_lang = payload.get("source", "auto")
                api_url = (
                    "https://translate.googleapis.com/translate_a/single?client=gtx"
                    f"&sl={source_lang}&tl={target_lang}&dt=t&q="
                    + urllib.parse.quote(text)
                )
                req = urllib.request.Request(
                    api_url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    translated_segments = [s[0] for s in resp_data[0] if s and s[0]]
                    translated = "".join(translated_segments)

                self.send_json_response({
                    "original": text,
                    "translated": translated,
                    "target_lang": target_lang
                })
            except Exception as e:
                self.send_json_response({"error": f"Translation failed: {str(e)}"}, status=500)
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
            if not wav_name.endswith(".wav"):
                wav_name += ".wav"

            wav_path = os.path.join(WAVS_DIR, wav_name)
            if not os.path.exists(wav_path):
                wav_path = os.path.join(PROCESSED_WAVS_DIR, wav_name)

            if os.path.exists(wav_path):
                self.serve_audio_file(wav_path)
            else:
                self.send_error(404, f"Audio file not found: {wav_name}")
            return

        # Default: Serve static files from web/
        super().do_GET()

    def handle_api_stats(self):
        total_wavs = 0
        if os.path.exists(WAVS_DIR):
            total_wavs = sum(1 for name in os.listdir(WAVS_DIR) if name.lower().endswith(".wav"))
        elif os.path.exists(PROCESSED_WAVS_DIR):
            total_wavs = sum(1 for name in os.listdir(PROCESSED_WAVS_DIR) if name.lower().endswith(".wav"))

        stats = {
            "dataset_name": "Khmer Male Speech (km_kh_male)",
            "total_wavs": total_wavs,
            "sample_rate": 48000,
            "target_rate": 22050,
            "language": "Khmer (km-KH)",
            "models": {
                "acoustic_onnx": os.path.exists(os.path.join(WEB_DIR, "models", "acoustic.onnx")),
                "vocoder_onnx": os.path.exists(os.path.join(WEB_DIR, "models", "vocoder.onnx")),
                "acoustic_ckpt": os.path.exists(os.path.join(BASE_DIR, "checkpoints", "acoustic", "best_acoustic.pt")),
                "vocoder_ckpt": os.path.exists(os.path.join(BASE_DIR, "checkpoints", "vocoder", "best_vocoder.pt")),
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

        all_samples = []
        if os.path.exists(SAMPLES_JSON):
            with open(SAMPLES_JSON, encoding="utf-8") as f:
                all_samples = json.load(f)
        else:
            meta_csv = os.path.join(DATASET_DIR, "metadata.csv")
            if os.path.exists(meta_csv):
                with open(meta_csv, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("|", 1)
                        if len(parts) == 2:
                            all_samples.append({"id": parts[0].strip(), "text": parts[1].strip()})

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
        print(f" KHM-TtS Web Studio & Dataset Explorer")
        print(f" Running at: http://localhost:{PORT}")
        print(f" Web UI: {WEB_DIR}")
        print(f"============================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    run_server()
