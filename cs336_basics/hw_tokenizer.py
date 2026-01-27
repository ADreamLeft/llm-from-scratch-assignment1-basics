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
        # 构建反向词表: bytes -> int
        self.encoder = {v: k for k, v in vocab.items()}
        self.merges = merges
        # 构建合并优先级字典: (byte1, byte2) -> rank (index)
        # 越小的 rank 表示合并优先级越高（越早被学习到）
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.special_tokens = special_tokens if special_tokens is not None else []

        # 预编译特殊 Token 的正则，用于 split
        self.special_pattern = None
        if self.special_tokens:
            # 按长度降序排序，确保长 token 优先匹配 (例如 '<|end|>' 优于 '<|')
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            # 使用 capturing group () 保留分隔符
            self.special_pattern = re.compile(
                f"({'|'.join(re.escape(s) for s in sorted_specials)})"
            )

        # 简单的缓存，避免重复计算相同单词的 BPE
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
        # 假设 vocab 存储为 json: {"0": "hex_string" or "latin1_string"}
        # 这里为了通用性，假设存储时将 bytes decode 成了 latin-1 字符串以保留 0-255 值
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)

        # 恢复 vocab: int -> bytes
        vocab = {}
        for k, v in raw_vocab.items():
            # 假设 v 是用 latin-1 编码保存的字符串，还原为 bytes
            vocab[int(k)] = v.encode("latin-1")

        # 假设 merges 存储为 json: [["byte_str1", "byte_str2"], ...]
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

        # 初始状态：每个字节都是一个独立的 token
        word = [bytes([b]) for b in token_bytes]

        while len(word) > 1:
            # 1. 找出当前序列中所有相邻的 pair
            min_rank = float("inf")
            min_pair = None
            min_idx = -1

            # 遍历寻找 rank 最小（优先级最高）的 pair
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair
                    min_idx = i

            # 如果没有可合并的 pair，结束
            if min_pair is None or min_rank == float("inf"):
                break

            # 2. 执行合并
            # 注意：标准的 BPE 实现通常是一次性合并所有出现的该 pair，
            # 但为了简单和准确，这里我们只合并找到的那个位置，循环会再次处理其他位置。
            # (对于长文本虽然效率低，但对于单个单词 token 处理是可接受的)
            new_token = min_pair[0] + min_pair[1]
            word[min_idx] = new_token
            del word[min_idx + 1]

        # 映射回 IDs
        ids = [self.encoder[b] for b in word]
        self.cache[token_bytes] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        ids = []

        # 1. 处理 Special Tokens
        # 如果有特殊 token，用正则切分，保留分隔符
        if self.special_pattern:
            parts = self.special_pattern.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue

            # 如果是特殊 Token，直接查找 ID
            if part in self.special_tokens:
                # 特殊 Token 在 vocab 中以 utf-8 bytes 形式存在
                part_bytes = part.encode("utf-8")
                if part_bytes in self.encoder:
                    ids.append(self.encoder[part_bytes])
                else:
                    # 理论上不应发生，除非 special_tokens 列表和 vocab 不匹配
                    print(f"Warning: Special token {part} not in vocab")
                continue

            # 2. 对普通文本进行 GPT-2 预分词 (Pre-tokenization)
            # 使用 regex.findall 找到所有匹配的单词块
            pre_tokens = re.findall(GPT2_SPLIT_PATTERN, part)

            for pre_token in pre_tokens:
                token_bytes = pre_token.encode("utf-8")
                # 3. 对每个块应用 BPE
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
        # 1. 将 IDs 转换回 bytes
        byte_parts = []
        for i in ids:
            if i in self.vocab:
                byte_parts.append(self.vocab[i])
            else:
                # 遇到未知 ID 的回退策略，通常不应发生
                pass

        # 2. 拼接所有 bytes
        full_bytes = b"".join(byte_parts)

        # 3. 解码为字符串，使用 'replace' 替换无效的 utf-8 序列 (U+FFFD)
        return full_bytes.decode("utf-8", errors="replace")
