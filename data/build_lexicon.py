"""
Build a lexicon dictionary for client-side text-to-phoneme / text-to-token lookup.
Supports:
1. English datasets (CMUdict + G2P fallback for missing words)
2. Khmer / Character-based datasets (word-to-character / token mapping)

Usage:
  python build_lexicon.py --data_dir km_kh_male --out web/models/lexicon.json --vocab_out web/models/vocab.json
"""
import argparse
import json
import os
import re


def build_khmer_or_char_lexicon(data_dir: str):
    meta_csv = os.path.join(data_dir, "metadata.csv")
    meta_tsv = os.path.join(data_dir, "line_index.tsv")

    lines = []
    if os.path.exists(meta_csv):
        with open(meta_csv, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    elif os.path.exists(meta_tsv):
        with open(meta_tsv, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

    lexicon = {}
    vocab = {"<pad>": 0, "<unk>": 1, "<space>": 2}

    for line in lines:
        if "|" in line:
            text = line.split("|", 1)[1]
        elif "\t" in line:
            text = line.split("\t")[-1]
        else:
            continue

        words = text.split()
        for w in words:
            chars = list(w)
            for c in chars:
                if c not in vocab:
                    vocab[c] = len(vocab)
            if w not in lexicon:
                lexicon[w] = chars

    return lexicon, vocab


def build_english_lexicon(data_dir: str):
    import nltk
    from g2p_en import G2p

    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict
    cmu = cmudict.dict()

    lexicon = {}
    vocab = {"<pad>": 0, "<unk>": 1, "<space>": 2}

    for word, prons in cmu.items():
        phones = [re.sub(r"(\d)$", r"\1", p) for p in prons[0]]
        lexicon[word.lower()] = phones
        for p in phones:
            if p not in vocab:
                vocab[p] = len(vocab)

    meta_csv = os.path.join(data_dir, "metadata.csv")
    g2p = G2p()
    missing = set()
    if os.path.exists(meta_csv):
        with open(meta_csv, encoding="utf-8") as f:
            for line in f:
                if "|" not in line:
                    continue
                _, text = line.strip().split("|", 1)
                for w in re.findall(r"[a-zA-Z']+", text):
                    if w.lower() not in lexicon:
                        missing.add(w.lower())

    for w in missing:
        phones = [p for p in g2p(w) if p != " "]
        lexicon[w] = phones
        for p in phones:
            if p not in vocab:
                vocab[p] = len(vocab)

    return lexicon, vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="web/models/lexicon.json")
    ap.add_argument("--vocab_out", default="web/models/vocab.json")
    ap.add_argument("--mode", choices=["auto", "char", "english"], default="auto")
    args = ap.parse_args()

    # Determine mode
    mode = args.mode
    if mode == "auto":
        meta_csv = os.path.join(args.data_dir, "metadata.csv")
        meta_tsv = os.path.join(args.data_dir, "line_index.tsv")
        first_line = ""
        for p in (meta_csv, meta_tsv):
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    first_line = f.readline()
                break
        if any(ord(c) > 127 for c in first_line):
            mode = "char"
        else:
            mode = "english"

    if mode == "char":
        lexicon, vocab = build_khmer_or_char_lexicon(args.data_dir)
    else:
        lexicon, vocab = build_english_lexicon(args.data_dir)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(lexicon)} words to {args.out}")

    if args.vocab_out:
        os.makedirs(os.path.dirname(args.vocab_out), exist_ok=True)
        with open(args.vocab_out, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"wrote {len(vocab)} tokens to {args.vocab_out}")


if __name__ == "__main__":
    main()
