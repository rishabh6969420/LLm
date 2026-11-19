"""
Character & Subword Tokenizer with Location & Fine-Grained Shade Tokens for Strikler.
"""

from typing import List, Dict
import re
import torch

from data.color_utils import COLOR_SHADE_DATABASE

SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<image>",
    "<detect_all>",
    "<detect_color>",
    "<color_at>",
]


class VLMTokenizer:
    """
    Tokenizer supporting natural language, special task tokens,
    80+ fine-grained color shades, and 1000 discrete spatial coordinate tokens <loc_0> to <loc_999>.
    """

    def __init__(self, vocab_size: int = 32000, num_loc_tokens: int = 1000):
        self.vocab_size = vocab_size
        self.num_loc_tokens = num_loc_tokens

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # 1. Add special tokens
        for i, token in enumerate(SPECIAL_TOKENS):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        # 2. Add ASCII characters
        curr_id = len(SPECIAL_TOKENS)
        for char in " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?-+*/_#()[]{}<>=\n\"'":
            if char not in self.token_to_id:
                self.token_to_id[char] = curr_id
                self.id_to_token[curr_id] = char
                curr_id += 1

        # 3. Add common vision and color task words
        common_words = [
            "detect", "objects", "color", "colors", "shade", "shades",
            "circle", "square", "rectangle", "triangle", "ellipse", "polygon",
            "box", "background", "foreground", "location", "hex", "parent"
        ]
        # Add all shade names from database (e.g. turquoise, blood, wine, burgundy, cobalt)
        for shade in COLOR_SHADE_DATABASE.keys():
            for part in shade.split():
                if part not in common_words:
                    common_words.append(part)

        for word in common_words:
            if word not in self.token_to_id:
                self.token_to_id[word] = curr_id
                self.id_to_token[curr_id] = word
                curr_id += 1

        # 4. Location tokens start at vocab_size
        self.loc_offset = vocab_size
        for loc in range(num_loc_tokens):
            loc_token = f"<loc_{loc}>"
            self.token_to_id[loc_token] = self.loc_offset + loc
            self.id_to_token[self.loc_offset + loc] = loc_token

        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]

    def encode(self, text: str) -> List[int]:
        """Encode text containing regular words, shades, and special tokens like <loc_123>."""
        pattern = r"(<loc_\d+>|<[^>]+>|\w+|[^\w\s])"
        tokens = re.findall(pattern, text)
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                for char in token:
                    ids.append(self.token_to_id.get(char, self.token_to_id["<unk>"]))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode a sequence of token IDs into text."""
        result = []
        for tid in ids:
            if tid == self.pad_id:
                continue
            if tid in self.id_to_token:
                tok = self.id_to_token[tid]
                if tok.startswith("<loc_") or tok.startswith("<"):
                    result.append(tok)
                elif len(tok) == 1:
                    result.append(tok)
                else:
                    result.append(" " + tok)
            else:
                result.append("?")
        return "".join(result).strip()

    def coord_to_token(self, coord: int) -> str:
        """Convert a normalized coordinate [0, 999] to token string."""
        coord = max(0, min(self.num_loc_tokens - 1, int(coord)))
        return f"<loc_{coord}>"
