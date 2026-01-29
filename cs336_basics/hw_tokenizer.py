import json
import os
from collections.abc import Iterable, Iterator

import regex as re

# 获取 CPU 核心数，保留一个核心给主进程或系统
CPU = os.cpu_count() or 1
# GPT-2 分割模式 regex
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class TestTokenizer:
    """
    Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.encoder = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_pattern = None
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pattern = re.compile(
                f"({'|'.join(re.escape(s) for s in sorted_specials)})"
            )
        self.cache = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary
        and list of merges and (optionally) a list of special tokens.
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {}
        for k, v in raw_vocab.items():
            vocab[int(k)] = v.encode("latin-1")
        with open(merges_filepath, encoding="utf-8") as f:
            raw_merges = json.load(f)

        merges = []
        for p1, p2 in raw_merges:
            merges.append((p1.encode("latin-1"), p2.encode("latin-1")))

        return cls(vocab, merges, special_tokens)

    def _bpe(self, token_bytes: bytes) -> list[int]:
        """
        核心 BPE 算法：将一个单词的 bytes 序列不断合并。
        返回该单词对应的 token IDs 列表。
        """
        if token_bytes in self.cache:
            return self.cache[token_bytes]
        word = [bytes([b]) for b in token_bytes]

        while len(word) > 1:
            min_rank = float("inf")
            min_pair = None
            min_idx = -1
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair
                    min_idx = i
            if min_pair is None or min_rank == float("inf"):
                break
            new_token = min_pair[0] + min_pair[1]
            word[min_idx] = new_token
            del word[min_idx + 1]
        ids = [self.encoder[b] for b in word]
        self.cache[token_bytes] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        ids = []
        if self.special_pattern:
            parts = self.special_pattern.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue

            if part in self.special_tokens:
                part_bytes = part.encode("utf-8")
                ids.append(self.encoder[part_bytes])
                continue

            pre_tokens = re.findall(GPT2_SPLIT_PATTERN, part)

            for pre_token in pre_tokens:
                token_bytes = pre_token.encode("utf-8")
                ids.extend(self._bpe(token_bytes))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings, return a generator that lazily yields token IDs.
        Suitable for processing large files line by line.
        """
        for text_chunk in iterable:
            token_ids = self.encode(text_chunk)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        byte_parts = []
        for i in ids:
            byte_parts.append(self.vocab[i])
        full_bytes = b"".join(byte_parts)
        return full_bytes.decode("utf-8", errors="replace")
