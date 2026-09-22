"""
Khmer Tokenizer for TTS:
Maps normalized Khmer text into token sequences and token IDs.
"""
import json
import os
from typing import Dict, List, Optional
from data.khmer_normalizer import KhmerNormalizer


class KhmerTokenizer:
    def __init__(self, vocab: Optional[Dict[str, int]] = None, vocab_path: Optional[str] = None):
        self.normalizer = KhmerNormalizer()

        if isinstance(vocab, str) and vocab_path is None:
            vocab_path = vocab
            vocab = None

        if vocab is not None:
            self.vocab = vocab
        elif vocab_path is not None and os.path.exists(vocab_path):
            with open(vocab_path, encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = {"<pad>": 0, "<unk>": 1, "<space>": 2, ",": 3, ".": 4, "?": 5, "!": 6}

        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text: str) -> List[str]:
        """Normalize text and split into graphemes/characters and pause tokens."""
        text = self.normalizer.normalize(text)
        words = text.split()
        tokens = []

        for i, word in enumerate(words):
            if word in (",", ".", "?", "!", ":"):
                tokens.append(word)
            else:
                for ch in word:
                    tokens.append(ch)

            if i < len(words) - 1 and word not in (",", ".", "?", "!", ":") and words[i + 1] not in (",", ".", "?", "!", ":"):
                tokens.append("<space>")

        return tokens

    def text_to_ids(self, text: str) -> List[int]:
        """Convert raw Khmer text directly into integer token IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(t, self.vocab.get("<unk>", 1)) for t in tokens]

    def encode(self, text: str) -> List[int]:
        """Standard alias for text_to_ids."""
        return self.text_to_ids(text)

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert integer token IDs back to token strings."""
        return [self.id_to_token.get(i, "<unk>") for i in ids]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back into text."""
        tokens = self.ids_to_tokens(ids)
        chars = []
        for t in tokens:
            if t == "<space>":
                chars.append(" ")
            elif not t.startswith("<"):
                chars.append(t)
        return "".join(chars)

    def build_vocab_from_texts(self, texts: List[str]) -> Dict[str, int]:
        """Build and update vocabulary from a list of Khmer texts."""
        for text in texts:
            tokens = self.tokenize(text)
            for t in tokens:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        return self.vocab

    def save_vocab(self, path: str):
        """Save vocabulary to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    tokenizer = KhmerTokenizer()
    sample = "ស្ពាន កំពង់ ចម្លង អ្នកលឿង"
    tokenizer.build_vocab_from_texts([sample])
    tokens = tokenizer.tokenize(sample)
    ids = tokenizer.text_to_ids(sample)
    print("Tokens:", tokens)
    print("IDs:", ids)
    print("Reconstructed:", tokenizer.ids_to_tokens(ids))
