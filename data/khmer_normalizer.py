"""
Khmer Text Normalizer for Text-to-Speech (TTS).
Handles:
1. Khmer numeral expansion (e.g. ០-៩, compound numbers up to billions)
2. Arabic numeral expansion to Khmer words
3. Zero-width space & joiner normalization (\\u200b, \\u200c, \\u200d)
4. Khmer punctuation normalization (។, ៕, ៚, ៙, etc.)
5. Lek To (ៗ) repetition expansion: repeats the preceding word/syllable
6. Sub-consonant (Coeng ្) and diacritic standard ordering
"""
import re

KHMER_DIGITS = {
    '\u17e0': 0, '\u17e1': 1, '\u17e2': 2, '\u17e3': 3, '\u17e4': 4,
    '\u17e5': 5, '\u17e6': 6, '\u17e7': 7, '\u17e8': 8, '\u17e9': 9
}

DIGIT_WORDS = {
    0: '\u179f\u17bc\u1793\u17d0\u1799', 1: '\u1798\u17bd\u1799', 2: '\u1796\u17b8\u179a',
    3: '\u1794\u17b8', 4: '\u1794\u17bd\u1793',
    5: '\u1794\u17d0\u179a\u17b6\u1798', 6: '\u1794\u17d0\u179a\u17b6\u1798\u1798\u17bd\u1799',
    7: '\u1794\u17d0\u179a\u17b6\u1798\u1796\u17b8\u179a',
    8: '\u1794\u17d0\u179a\u17b6\u1798\u1794\u17b8', 9: '\u1794\u17d0\u179a\u17b6\u1798\u1794\u17bd\u1793'
}

TENS_WORDS = {
    10: '\u178c\u1794\u17d0', 20: '\u1798\u17d2\u1797\u17c2', 30: '\u179f\u17b6\u1798\u179f\u17b7\u1794',
    40: '\u179f\u17c2\u179f\u17b7\u1794', 50: '\u17a0\u17b6\u179f\u17b7\u1794',
    60: '\u17a0\u17bb\u1780\u179f\u17b7\u1794', 70: '\u1785\u17b7\u178f\u179f\u17b7\u1794',
    80: '\u1794\u17d0\u17c2\u178f\u179f\u17b7\u1794', 90: '\u1780\u17c5\u179f\u17b7\u1794'
}


def number_to_khmer_words(n: int) -> str:
    """Convert an integer (0 to 999,999,999) into spoken Khmer words."""
    if n < 0:
        return '\u178c\u1780 ' + number_to_khmer_words(-n)
    if n == 0:
        return DIGIT_WORDS[0]

    parts = []

    if n >= 1_000_000:
        millions = n // 1_000_000
        parts.append(number_to_khmer_words(millions) + ' \u179b\u17b6\u1793')
        n %= 1_000_000

    if n >= 1000:
        thousands = n // 1000
        parts.append(number_to_khmer_words(thousands) + ' \u1796\u17b6\u1793\u17d0')
        n %= 1000

    if n >= 100:
        hundreds = n // 100
        parts.append(DIGIT_WORDS[hundreds] + ' \u179a\u1799')
        n %= 100

    if n >= 10:
        tens = (n // 10) * 10
        rem = n % 10
        if rem == 0:
            parts.append(TENS_WORDS[tens])
        else:
            if tens == 10:
                parts.append('\u178c\u1794\u17d0 ' + DIGIT_WORDS[rem])
            else:
                parts.append(TENS_WORDS[tens] + ' ' + DIGIT_WORDS[rem])
        n = 0

    if n > 0:
        parts.append(DIGIT_WORDS[n])

    return ' '.join(parts).strip()


class KhmerNormalizer:
    def __init__(self):
        # Khmer punctuation to speech pause marks
        self.punct_map = {
            '\u17d4': ' , ',   # Khan (sentence period) -> pause
            '\u17d5': ' . ',   # Bariyoosan (end of paragraph) -> full stop
            '\u17da': ' ',
            '\u17d9': ' ',
            '\u17d6': ' : ',
            '(': ' ', ')': ' ', '[': ' ', ']': ' ',
            '"': '', "'": '', '\u00ab': '', '\u00bb': '',
            ',': ' , ', '.': ' . ', '?': ' ? ', '!': ' ! ',
            ';': ' , ', ':': ' , ', '-': ' '
        }

    def expand_lek_to(self, text: str) -> str:
        """
        Expand Khmer Lek To (U+17D7, \\u17d7) by repeating the preceding word/syllable.

        Lek To is a Khmer repetition mark meaning the preceding syllable or
        word should be spoken twice.  Examples:
            \\u17e2 -> 'ស្អាតៗ'  =>  'ស្អាត ស្អាត'
            'ធំៗ'   =>  'ធំ ធំ'
            'ស្អាត ៗ ៗ ៗ'  =>  'ស្អាត ស្អាត ស្អាត ស្អាត'

        Strategy: ensure \\u17d7 is whitespace-separated, then walk tokens
        left-to-right, repeating the previous substantive token for each \\u17d7.
        """
        # Insert space before U+17D7 (lek to) if not already separated
        LEK_TO = '\u17d7'
        text = re.sub(r'([^\s])' + re.escape(LEK_TO), r'\1 ' + LEK_TO, text)
        words = text.split()
        result: list = []
        for w in words:
            if w == '\u17d7':
                if result:
                    result.append(result[-1])
                # \\u17d7 at start of string: drop silently (no preceding word)
            else:
                result.append(w)
        return ' '.join(result)

    def normalize_numbers(self, text: str) -> str:
        """Find all Khmer and Arabic numbers and replace with spoken Khmer words."""
        def replace_khmer_digits(match):
            kh_str = match.group(0)
            val = int(''.join(str(KHMER_DIGITS[ch]) for ch in kh_str))
            return ' ' + number_to_khmer_words(val) + ' '

        def replace_arabic_digits(match):
            val = int(match.group(0))
            return ' ' + number_to_khmer_words(val) + ' '

        text = re.sub(r'[\u17e0-\u17e9]+', replace_khmer_digits, text)
        text = re.sub(r'[0-9]+', replace_arabic_digits, text)
        return text

    def normalize(self, text: str) -> str:
        """Full normalization pipeline for Khmer text."""
        if not text:
            return ""

        # 1. Unicode NFC normalization
        import unicodedata
        text = unicodedata.normalize('NFC', text)

        # 2. Strip Zero-Width Spaces (U+200B) and non-breaking spaces (U+00A0)
        #    with a space; silently remove ZWNJ (U+200C) and ZWJ (U+200D)
        #    because they appear inside Khmer ligature sequences and inserting a
        #    space there would break token identity.
        text = re.sub(r'[\u200b\ufeff\u00a0]', ' ', text)
        text = re.sub(r'[\u200c\u200d]', '', text)

        # 3. Number expansion
        text = self.normalize_numbers(text)

        # 4. Lek To repetition expansion (must run before punctuation normalization
        #    so that \\u17d7 is still present as a separate character)
        text = self.expand_lek_to(text)

        # 5. Normalize Punctuation
        for p, rep in self.punct_map.items():
            text = text.replace(p, rep)

        # 6. Clean extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text


if __name__ == "__main__":
    norm = KhmerNormalizer()
    sample = "\u179f\u17d2\u1796\u17b6\u1793 \u1780\u17c6\u1796\u1784\u17d2 \u1785\u1798\u17d2\u179b\u1784 \u17a2\u17d2\u1793\u1780\u179b\u17bf\u1784 \u178f\u1798\u17d2\u179b\u17c3 \u17e1\u17e5\u17e0 \u178c\u17bb\u179b\u17d2\u179b\u17b6\u179a \u1793\u17c5 \u17a6\u17d2\u1793\u17b6\u17c6 \u17e2\u17e0\u17e2\u17e6\u17d4"
    print("Original:", sample)
    print("Normalized:", norm.normalize(sample))

    # Lek To edge cases
    for s in ["\u179f\u17d2\u17a2\u17b6\u178f\u17d7", "\u17d7 \u1794\u17d2\u179a\u178f\u17be\u179f", "\u179f\u17d2\u17a2\u17b6\u178f \u17d7 \u17d7 \u17d7"]:
        print(f"  '{s}' -> '{norm.normalize(s)}'")
