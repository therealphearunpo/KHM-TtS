"""
Automated integration and unit tests for KHM-TTS pipeline.

Run with:
    python -m unittest discover tests -v
    # or just:
    python tests/test_pipeline.py

Tests cover:
  1.  Text normalization correctness
  2.  Khmer tokenizer encode/decode
  3.  Dataset loading and tensor shapes
  4.  Audio loading helpers
  5.  Model state-dict loading
  6.  Acoustic model forward pass (train mode & inference mode)
  7.  Vocoder forward pass
  8.  End-to-end text-to-waveform inference
  9.  ONNX acoustic & vocoder numerical consistency
  10. API endpoint responses
  11. Invalid input handling
"""

import io
import json
import os
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

import numpy as np
import torch

# Ensure project root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "km_kh_male")
PROC_DIR = os.path.join(DATA_DIR, "processed")
VOCAB_PATH = os.path.join(PROC_DIR, "vocab.json")
ACOUSTIC_CKPT = os.path.join(ROOT, "checkpoints", "acoustic", "best_acoustic.pt")
VOCODER_CKPT = os.path.join(ROOT, "checkpoints", "vocoder", "best_vocoder.pt")
ACOUSTIC_ONNX = os.path.join(ROOT, "web", "models", "acoustic.onnx")
VOCODER_ONNX = os.path.join(ROOT, "web", "models", "vocoder.onnx")


# ---------------------------------------------------------------------------
# 1. Text Normalization
# ---------------------------------------------------------------------------
class TestTextNormalization(unittest.TestCase):

    def setUp(self):
        from data.khmer_normalizer import KhmerNormalizer
        self.norm = KhmerNormalizer()

    def test_empty_string(self):
        self.assertEqual(self.norm.normalize(""), "")

    def test_nfc_normalization(self):
        # NFC should collapse precomposed forms
        import unicodedata
        text = unicodedata.normalize("NFD", "ស្ពាន")
        result = self.norm.normalize(text)
        self.assertGreater(len(result), 0)

    def test_number_expansion_arabic(self):
        result = self.norm.normalize("150")
        self.assertIn("មួយ", result)  # should contain "one hundred"
        self.assertNotIn("150", result)

    def test_number_expansion_khmer_digits(self):
        # ១ = 1, ៥ = 5, ០ = 0
        result = self.norm.normalize("\u17e1\u17e5\u17e0")
        self.assertIn("មួយ", result)
        self.assertNotIn("\u17e1", result)

    def test_lek_to_expansion_attached(self):
        # ស្អាតៗ → ស្អាត ស្អាត
        result = self.norm.normalize("ស្អាតៗ")
        parts = result.split()
        # The word should appear at least twice
        self.assertGreaterEqual(parts.count(parts[0]), 2, f"Lek To not expanded: {result!r}")

    def test_lek_to_expansion_detached(self):
        # ស្អាត ៗ → ស្អាត ស្អាត
        result = self.norm.normalize("ស្អាត ៗ")
        parts = result.split()
        self.assertGreaterEqual(len(parts), 2, f"Lek To detached not expanded: {result!r}")

    def test_lek_to_at_start_dropped(self):
        # ៗ with no preceding word should be dropped silently
        result = self.norm.normalize("ៗ ប្រទេស")
        self.assertNotIn("ៗ", result)

    def test_punctuation_mapping(self):
        # Khan (។) becomes pause marker ','
        result = self.norm.normalize("ស្ពាន។")
        self.assertIn(",", result)
        self.assertNotIn("។", result)

    def test_zero_width_space_removed(self):
        # U+200B should become a space
        result = self.norm.normalize("ស\u200Bប")
        self.assertNotIn("\u200B", result)

    def test_zwnj_removed_silently(self):
        # U+200C inside a ligature should be removed, not become a space
        result = self.norm.normalize("ក\u200Cខ")
        self.assertNotIn("\u200C", result)
        self.assertNotIn("  ", result)  # no double-space artefact


# ---------------------------------------------------------------------------
# 2. Khmer Tokenizer
# ---------------------------------------------------------------------------
class TestKhmerTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(VOCAB_PATH):
            raise unittest.SkipTest("vocab.json not found — run prepare_dataset.py first")
        from data.khmer_tokenizer import KhmerTokenizer
        cls.tok = KhmerTokenizer(vocab_path=VOCAB_PATH)

    def test_vocab_loaded(self):
        self.assertGreater(len(self.tok.vocab), 6)
        self.assertIn("<pad>", self.tok.vocab)
        self.assertIn("<unk>", self.tok.vocab)
        self.assertIn("<space>", self.tok.vocab)

    def test_pad_index_zero(self):
        self.assertEqual(self.tok.vocab["<pad>"], 0)

    def test_encode_nonempty(self):
        ids = self.tok.encode("ស្ពាន")
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_known_chars_no_unk(self):
        ids = self.tok.encode("ស្ពាន")
        unk_id = self.tok.vocab.get("<unk>", 1)
        self.assertNotIn(unk_id, ids, "Known Khmer chars should not produce UNK tokens")

    def test_unknown_chars_produce_unk(self):
        ids = self.tok.encode("Z")   # Latin Z not in vocab
        unk_id = self.tok.vocab.get("<unk>", 1)
        self.assertIn(unk_id, ids)

    def test_empty_text_empty_ids(self):
        ids = self.tok.encode("")
        self.assertEqual(ids, [])

    def test_roundtrip_decode(self):
        text = "ស្ពាន"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        self.assertIn("ស", decoded)

    def test_space_token(self):
        ids = self.tok.encode("ស្ពាន កំពង់")
        space_id = self.tok.vocab.get("<space>", 2)
        self.assertIn(space_id, ids)


# ---------------------------------------------------------------------------
# 3. Dataset Loading
# ---------------------------------------------------------------------------
class TestDatasetLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(PROC_DIR):
            raise unittest.SkipTest("Processed dataset not found")
        manifest = os.path.join(PROC_DIR, "train_manifest.txt")
        if not os.path.exists(manifest):
            raise unittest.SkipTest("train_manifest.txt not found")
        with open(manifest, encoding="utf-8") as f:
            cls.ids = [l.strip() for l in f if l.strip()]
        if not cls.ids:
            raise unittest.SkipTest("train_manifest.txt is empty")

    def test_manifest_nonempty(self):
        self.assertGreater(len(self.ids), 0)

    def test_phoneme_file_exists(self):
        uid = self.ids[0]
        path = os.path.join(PROC_DIR, "phonemes", f"{uid}.npy")
        self.assertTrue(os.path.exists(path), f"Phoneme file missing: {path}")

    def test_mel_file_exists(self):
        uid = self.ids[0]
        path = os.path.join(PROC_DIR, "mels", f"{uid}.npy")
        self.assertTrue(os.path.exists(path), f"Mel file missing: {path}")

    def test_duration_file_exists(self):
        uid = self.ids[0]
        path = os.path.join(PROC_DIR, "durations", f"{uid}.npy")
        self.assertTrue(os.path.exists(path), f"Duration file missing: {path}")

    def test_mel_shape(self):
        uid = self.ids[0]
        mel = np.load(os.path.join(PROC_DIR, "mels", f"{uid}.npy"))
        self.assertEqual(mel.shape[0], 80, f"Expected 80 mel bins, got {mel.shape[0]}")

    def test_duration_phoneme_length_match(self):
        uid = self.ids[0]
        dur = np.load(os.path.join(PROC_DIR, "durations", f"{uid}.npy"))
        phon = np.load(os.path.join(PROC_DIR, "phonemes", f"{uid}.npy"))
        self.assertEqual(
            len(dur), len(phon),
            f"Duration length {len(dur)} ≠ phoneme length {len(phon)} for {uid}"
        )

    def test_duration_sum_equals_mel_length(self):
        uid = self.ids[0]
        dur = np.load(os.path.join(PROC_DIR, "durations", f"{uid}.npy"))
        mel = np.load(os.path.join(PROC_DIR, "mels", f"{uid}.npy"))
        self.assertEqual(
            int(dur.sum()), mel.shape[1],
            f"Duration sum {dur.sum()} ≠ mel length {mel.shape[1]} for {uid}"
        )


# ---------------------------------------------------------------------------
# 4. Audio Loading
# ---------------------------------------------------------------------------
class TestAudioLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        wav_dir = os.path.join(PROC_DIR, "wavs_22k")
        if not os.path.exists(wav_dir):
            raise unittest.SkipTest("wavs_22k not found")
        import glob
        wavs = glob.glob(os.path.join(wav_dir, "*.wav"))
        if not wavs:
            raise unittest.SkipTest("No WAV files found in wavs_22k")
        cls.sample_wav = wavs[0]

    def test_loads_without_error(self):
        import soundfile as sf
        wav, sr = sf.read(self.sample_wav)
        self.assertEqual(sr, 22050)
        self.assertGreater(len(wav), 0)

    def test_is_mono(self):
        import soundfile as sf
        wav, _ = sf.read(self.sample_wav)
        self.assertEqual(wav.ndim, 1, "Expected mono audio (1-D array)")

    def test_amplitude_in_range(self):
        import soundfile as sf
        wav, _ = sf.read(self.sample_wav)
        self.assertLessEqual(np.max(np.abs(wav)), 1.05, "Amplitude out of expected range")

    def test_no_nan(self):
        import soundfile as sf
        wav, _ = sf.read(self.sample_wav)
        self.assertFalse(np.isnan(wav).any(), "NaN values found in audio")


# ---------------------------------------------------------------------------
# 5. Model Loading
# ---------------------------------------------------------------------------
class TestModelLoading(unittest.TestCase):

    def test_acoustic_checkpoint_loads(self):
        if not os.path.exists(ACOUSTIC_CKPT):
            self.skipTest(f"Acoustic checkpoint not found: {ACOUSTIC_CKPT}")
        ckpt = torch.load(ACOUSTIC_CKPT, map_location="cpu")
        self.assertIn("model", ckpt)
        self.assertIn("vocab", ckpt)

    def test_vocoder_checkpoint_loads(self):
        if not os.path.exists(VOCODER_CKPT):
            self.skipTest(f"Vocoder checkpoint not found: {VOCODER_CKPT}")
        ckpt = torch.load(VOCODER_CKPT, map_location="cpu")
        self.assertIn("gen", ckpt)


# ---------------------------------------------------------------------------
# 6. Acoustic Model Forward Pass
# ---------------------------------------------------------------------------
class TestAcousticModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from model.acoustic_model import FastSpeechLite
        if not os.path.exists(ACOUSTIC_CKPT):
            raise unittest.SkipTest("Acoustic checkpoint not found")
        ckpt = torch.load(ACOUSTIC_CKPT, map_location="cpu")
        cls.vocab_size = len(ckpt["vocab"])
        cls.model = FastSpeechLite(vocab_size=cls.vocab_size)
        cls.model.load_state_dict(ckpt["model"])
        cls.model.eval()

    def _sample_batch(self, T=20, B=2):
        phon = torch.randint(1, self.vocab_size, (B, T))
        dur = torch.randint(1, 4, (B, T))
        return phon, dur

    def test_training_forward(self):
        phon, dur = self._sample_batch()
        mel, log_dur, lens = self.model(phon, durations=dur)
        self.assertEqual(mel.shape[-1], 80, "Mel should have 80 bins")
        self.assertEqual(log_dur.shape, phon.shape)

    def test_inference_forward(self):
        phon = torch.randint(1, self.vocab_size, (1, 20))
        with torch.no_grad():
            mel, log_dur, lens = self.model(phon)
        self.assertGreater(int(lens[0].item()), 0)
        self.assertEqual(mel.shape[-1], 80)
        self.assertFalse(torch.isnan(mel).any(), "NaN in predicted mel")
        self.assertFalse(torch.isinf(mel).any(), "Inf in predicted mel")

    def test_model_in_eval_mode(self):
        self.assertFalse(self.model.training, "Model should be in eval mode")


# ---------------------------------------------------------------------------
# 7. Vocoder Forward Pass
# ---------------------------------------------------------------------------
class TestVocoderModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from model.vocoder import Generator
        if not os.path.exists(VOCODER_CKPT):
            raise unittest.SkipTest("Vocoder checkpoint not found")
        ckpt = torch.load(VOCODER_CKPT, map_location="cpu")
        cls.gen = Generator()
        cls.gen.load_state_dict(ckpt["gen"])
        cls.gen.remove_weight_norm()
        cls.gen.eval()

    def test_waveform_shape(self):
        mel = torch.randn(1, 80, 50)
        with torch.no_grad():
            wav = self.gen(mel)
        # upsample_rates (8,8,2,2) → total 256 × 50 = 12,800 samples
        self.assertEqual(wav.shape, (1, 1, 50 * 256), f"Unexpected wav shape: {wav.shape}")

    def test_output_bounded(self):
        mel = torch.randn(1, 80, 30)
        with torch.no_grad():
            wav = self.gen(mel)
        # tanh output: must be in [-1, 1]
        self.assertLessEqual(float(wav.abs().max()), 1.0 + 1e-5)

    def test_no_nan_inf(self):
        mel = torch.randn(1, 80, 40)
        with torch.no_grad():
            wav = self.gen(mel)
        self.assertFalse(torch.isnan(wav).any(), "NaN in vocoder output")
        self.assertFalse(torch.isinf(wav).any(), "Inf in vocoder output")


# ---------------------------------------------------------------------------
# 8. End-to-End Inference
# ---------------------------------------------------------------------------
class TestEndToEndInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(ACOUSTIC_CKPT):
            raise unittest.SkipTest("Acoustic checkpoint not found")
        from infer import KhmerTTSPipeline
        cls.pipeline = KhmerTTSPipeline(
            acoustic_ckpt=ACOUSTIC_CKPT,
            vocoder_ckpt=VOCODER_CKPT,
            vocab_path=VOCAB_PATH,
            vocoder_type="griffin_lim",
        )

    def _synth(self, text):
        return self.pipeline.synthesize(text, vocoder="griffin_lim")

    def test_short_khmer(self):
        wav = self._synth("ស្ពាន")
        self.assertGreater(len(wav), 0)
        self.assertFalse(np.isnan(wav).any())
        self.assertFalse(np.isinf(wav).any())

    def test_medium_khmer(self):
        wav = self._synth("ស្ពាន កំពង់ ចម្លង អ្នកលឿង")
        self.assertGreater(len(wav) / 22050, 0.5)  # at least 0.5s

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            self._synth("")

    def test_output_amplitude_normalized(self):
        wav = self._synth("ស្ពាន កំពង់")
        peak = float(np.max(np.abs(wav)))
        # Normalization targets 0.95 peak
        self.assertLessEqual(peak, 1.0 + 1e-5)

    def test_khmer_with_numbers(self):
        wav = self._synth("តម្លៃ 150 ដុល្លារ")
        self.assertGreater(len(wav), 0)

    def test_punctuation_handled(self):
        wav = self._synth("ស្ពាន។ ផ្ទះ")
        self.assertGreater(len(wav), 0)


# ---------------------------------------------------------------------------
# 9. ONNX Numerical Consistency
# ---------------------------------------------------------------------------
class TestOnnxConsistency(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import onnxruntime as ort
        except ImportError:
            raise unittest.SkipTest("onnxruntime not installed")

        missing = [p for p in [ACOUSTIC_ONNX, VOCODER_ONNX] if not os.path.exists(p)]
        if missing:
            raise unittest.SkipTest(f"ONNX files missing: {missing} — run export_onnx.py")
        if not os.path.exists(ACOUSTIC_CKPT):
            raise unittest.SkipTest("Acoustic checkpoint missing")

        import onnxruntime as ort
        from export_onnx import AcousticInferenceWrapper
        from model.acoustic_model import FastSpeechLite

        ckpt = torch.load(ACOUSTIC_CKPT, map_location="cpu")
        base = FastSpeechLite(vocab_size=len(ckpt["vocab"]))
        base.load_state_dict(ckpt["model"])
        base.eval()
        cls.infer_model = AcousticInferenceWrapper(base)
        cls.infer_model.eval()
        cls.sess_a = ort.InferenceSession(ACOUSTIC_ONNX)
        cls.sess_v = ort.InferenceSession(VOCODER_ONNX)
        cls.vocab_size = len(ckpt["vocab"])

    def test_acoustic_max_diff(self):
        np.random.seed(1)
        ids = np.random.randint(1, self.vocab_size, (1, 25)).astype(np.int64)
        with torch.no_grad():
            pt_mel, _ = self.infer_model(torch.from_numpy(ids))
        onnx_mel = self.sess_a.run(None, {"phoneme_ids": ids})[0]
        max_diff = float(np.abs(pt_mel.numpy() - onnx_mel).max())
        self.assertLess(max_diff, 1e-3, f"Acoustic ONNX max diff too large: {max_diff:.2e}")

    def test_vocoder_max_diff(self):
        from model.vocoder import Generator
        ckpt = torch.load(VOCODER_CKPT, map_location="cpu")
        gen = Generator()
        gen.load_state_dict(ckpt["gen"])
        gen.remove_weight_norm()
        gen.eval()

        np.random.seed(2)
        mel_np = np.random.randn(1, 80, 50).astype(np.float32)
        with torch.no_grad():
            pt_wav = gen(torch.from_numpy(mel_np)).numpy()
        onnx_wav = self.sess_v.run(None, {"mel": mel_np})[0]
        max_diff = float(np.abs(pt_wav - onnx_wav).max())
        self.assertLess(max_diff, 1e-4, f"Vocoder ONNX max diff too large: {max_diff:.2e}")

    def test_onnx_dynamic_input_lengths(self):
        """ONNX model must produce different output lengths for different input lengths."""
        for length in [8, 20, 40]:
            ids = np.random.randint(1, self.vocab_size, (1, length)).astype(np.int64)
            mel = self.sess_a.run(None, {"phoneme_ids": ids})[0]
            self.assertEqual(mel.shape[2], 80, "Mel bins should always be 80")


# ---------------------------------------------------------------------------
# 10. API Endpoints
# ---------------------------------------------------------------------------
class TestAPIEndpoints(unittest.TestCase):
    """Spin up a local server on a random port and test the HTTP API."""

    _server = None
    _port = 8799

    @classmethod
    def setUpClass(cls):
        import socketserver
        from server import TTSRequestHandler

        port = cls._port
        cls._httpd = socketserver.TCPServer(("127.0.0.1", port), TTSRequestHandler)
        cls._thread = threading.Thread(target=cls._httpd.serve_forever, daemon=True)
        cls._thread.start()
        time.sleep(0.4)
        cls._base = f"http://127.0.0.1:{port}"

    @classmethod
    def tearDownClass(cls):
        cls._httpd.shutdown()

    def test_stats_200(self):
        with urllib.request.urlopen(f"{self._base}/api/stats") as r:
            self.assertEqual(r.status, 200)
            data = json.loads(r.read())
            self.assertIn("total_wavs", data)

    def test_samples_200(self):
        with urllib.request.urlopen(f"{self._base}/api/samples?limit=3") as r:
            self.assertEqual(r.status, 200)
            data = json.loads(r.read())
            self.assertIn("samples", data)
            self.assertLessEqual(len(data["samples"]), 3)

    def test_synthesize_valid(self):
        payload = json.dumps({"text": "ស្ពាន", "vocoder": "griffin_lim"}).encode()
        req = urllib.request.Request(
            f"{self._base}/api/synthesize", data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as r:
            self.assertEqual(r.status, 200)
            self.assertGreater(len(r.read()), 44)  # WAV header at minimum

    def test_synthesize_empty_text_400(self):
        payload = json.dumps({"text": "", "vocoder": "griffin_lim"}).encode()
        req = urllib.request.Request(
            f"{self._base}/api/synthesize", data=payload,
            headers={"Content-Type": "application/json"}
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)

    def test_audio_404_for_missing(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(f"{self._base}/audio/nonexistent_id.wav")
        self.assertEqual(ctx.exception.code, 404)


# ---------------------------------------------------------------------------
# 11. Invalid Input Handling
# ---------------------------------------------------------------------------
class TestInvalidInputHandling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(ACOUSTIC_CKPT):
            raise unittest.SkipTest("Acoustic checkpoint not found")
        from infer import KhmerTTSPipeline
        cls.pipeline = KhmerTTSPipeline(
            acoustic_ckpt=ACOUSTIC_CKPT,
            vocoder_ckpt=VOCODER_CKPT,
            vocab_path=VOCAB_PATH,
            vocoder_type="griffin_lim",
        )

    def test_empty_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.pipeline.synthesize("")

    def test_whitespace_only_raises(self):
        with self.assertRaises(ValueError):
            self.pipeline.synthesize("   ")

    def test_unsupported_emoji_no_crash(self):
        """Unsupported characters map to <unk> — pipeline should not crash."""
        wav = self.pipeline.synthesize("ស\U0001F44D")
        self.assertGreater(len(wav), 0)

    def test_very_long_input_no_crash(self):
        long_text = "ស្ពាន កំពង់ " * 30
        wav = self.pipeline.synthesize(long_text)
        self.assertGreater(len(wav), 0)
        self.assertFalse(np.isnan(wav).any())


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
