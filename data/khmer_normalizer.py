"""
Khmer Text Normalizer for Text-to-Speech (TTS).
Handles:
1. Khmer numeral expansion (e.g. ០-៩, compound numbers up to billions)
2. Arabic numeral expansion to Khmer words
3. Khmer repetition mark (Lek To ៗ) word expansion
4. Zero-width space & joiner normalization (\u200b, \u200c, \u200d)
5. Khmer punctuation normalization (។, ៕, ៚, ៙, etc.)
"""
import re
import unicodedata

KHMER_DIGITS = {
    '០': 0, '១': 1, '២': 2, '៣': 3, '៤': 4,
    '៥': 5, '៦': 6, '៧': 7, '៨': 8, '៩': 9
}

DIGIT_WORDS = {
    0: 'សូន្យ', 1: 'មួយ', 2: 'ពីរ', 3: 'បី', 4: 'បួន',
    5: 'ប្រាំ', 6: 'ប្រាំមួយ', 7: 'ប្រាំពីរ', 8: 'ប្រាំបី', 9: 'ប្រាំបួន'
}

TENS_WORDS = {
    10: 'ដប់', 20: 'ម្ភៃ', 30: 'សាមសិប', 40: 'សែសិប',
    50: 'ហាសិប', 60: 'ហុកសិប', 70: 'ចិតសិប', 80: 'ប៉ែតសិប', 90: 'កៅសិប'
}


def number_to_khmer_words(n: int) -> str:
    """Convert an integer (0 to 999,999,999) into spoken Khmer words."""
    if n < 0:
        return 'ដក ' + number_to_khmer_words(-n)
    if n == 0:
        return DIGIT_WORDS[0]

    parts = []

    if n >= 1_000_000_000:
        billions = n // 1_000_000_000
        parts.append(number_to_khmer_words(billions) + ' ប៊ីលាន')
        n %= 1_000_000_000

    if n >= 1_000_000:
        millions = n // 1_000_000
        parts.append(number_to_khmer_words(millions) + ' លាន')
        n %= 1_000_000

    if n >= 1000:
        thousands = n // 1000
        parts.append(number_to_khmer_words(thousands) + ' ពាន់')
        n %= 1000

    if n >= 100:
        hundreds = n // 100
        parts.append(DIGIT_WORDS[hundreds] + ' រយ')
        n %= 100

    if n >= 10:
        tens = (n // 10) * 10
        rem = n % 10
        if rem == 0:
            parts.append(TENS_WORDS[tens])
        else:
            if tens == 10:
                parts.append('ដប់ ' + DIGIT_WORDS[rem])
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
            '។': ' , ',   # Khan (sentence period) -> pause
            '៕': ' . ',   # Bariyoosan (end of paragraph) -> full stop
            '៚': ' ',
            '៙': ' ',
            '៖': ' : ',
            '(': ' ', ')': ' ', '[': ' ', ']': ' ',
            '"': '', "'": '', '«': '', '»': '',
            ',': ' , ', '.': ' . ', '?': ' ? ', '!': ' ! ',
            ';': ' , ', ':': ' , ', '-': ' '
        }

    def expand_lek_to(self, text: str) -> str:
        """
        Expand Khmer repetition mark (ៗ / Lek To).
        Repeats the preceding Khmer word or syllable (e.g. 'ញឹកញាប់ៗ' -> 'ញឹកញាប់ ញឹកញាប់').
        """
        if 'ៗ' not in text:
            return text

        pattern = r'([\u1780-\u17DD]+)\s*ៗ'
        max_iter = 10
        while 'ៗ' in text and max_iter > 0:
            new_text = re.sub(pattern, r'\1 \1', text)
            if new_text == text:
                text = text.replace('ៗ', '')
                break
            text = new_text
            max_iter -= 1
        return text

    def normalize_numbers(self, text: str) -> str:
        """Find all Khmer and Arabic numbers and replace with spoken Khmer words."""
        def replace_khmer_digits(match):
            kh_str = match.group(0)
            val = int(''.join(str(KHMER_DIGITS[ch]) for ch in kh_str))
            return ' ' + number_to_khmer_words(val) + ' '

        def replace_arabic_digits(match):
            val = int(match.group(0))
            return ' ' + number_to_khmer_words(val) + ' '

        text = re.sub(r'[០-៩]+', replace_khmer_digits, text)
        text = re.sub(r'[0-9]+', replace_arabic_digits, text)
        return text

    def normalize(self, text: str) -> str:
        """Full normalization pipeline for Khmer text."""
        if not text:
            return ""

        # 1. Unicode NFC normalization
        text = unicodedata.normalize('NFC', text)

        # 2. Expand Khmer Lek To (ៗ) BEFORE punctuation mapping
        text = self.expand_lek_to(text)

        # 3. Handle zero-width characters properly:
        # \u200B (ZWSP) is word separator -> space
        # \u200C (ZWNJ) and \u200D (ZWJ) -> remove without space to avoid splitting ligatures
        # \uFEFF (BOM) -> remove
        # \u00A0 (NBSP) -> space
        text = text.replace('\u200B', ' ')
        text = text.replace('\u00A0', ' ')
        text = text.replace('\u200C', '')
        text = text.replace('\u200D', '')
        text = text.replace('\uFEFF', '')

        # 4. Number expansion
        text = self.normalize_numbers(text)

        # 5. Normalize Punctuation
        for p, rep in self.punct_map.items():
            text = text.replace(p, rep)

        # 6. Clean extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text


if __name__ == "__main__":
    norm = KhmerNormalizer()
    sample = "ស្ពាន កំពង់ ចម្លង អ្នកលឿង តម្លៃ ១៥០ ដុល្លារ នៅ ឆ្នាំ ២០២៦ ដើរលឿនៗ។"
    print("Original:", sample)
    print("Normalized:", norm.normalize(sample))
