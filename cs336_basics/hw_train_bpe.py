import heapq
import multiprocessing
import os
from collections import Counter, defaultdict

import regex as re

# 获取 CPU 核心数，保留一个核心给主进程或系统
CPU = os.cpu_count()
# GPT-2 分割模式 regex
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def find_chunk_boundaries(
    text: bytes,
    desired_num_chunks: int,
    split_special_tokens: list[str] | None = None,
) -> list[int]:
    length = len(text)

    if desired_num_chunks <= 1:
        return [0, length]

    chunk_size = length // desired_num_chunks
    chunk_boundaries = [0]

    # 预处理特殊 Token 的 bytes，用于在二进制流中查找
    special_tokens_bytes = []
    if split_special_tokens:
        special_tokens_bytes = [t.encode("utf-8") for t in split_special_tokens]

    for i in range(1, desired_num_chunks):
        target_position = i * chunk_size
        current_position = target_position

        # 读取 4KB 缓冲区寻找最佳切分点，防止读取整个文件导致内存爆炸
        search_window_size = 4096
        mini_chunk = text[current_position : current_position + search_window_size]

        if not mini_chunk:
            chunk_boundaries.append(length)
            break

        best_offset = -1

        # 1. 优先查找当前 buffer 中出现最早的任意特殊 Token
        if special_tokens_bytes:
            min_idx = search_window_size + 1
            for stb in special_tokens_bytes:
                idx = mini_chunk.find(stb)
                if idx != -1 and idx < min_idx:
                    min_idx = idx

            if min_idx <= search_window_size:
                best_offset = min_idx

        # 2. 如果没找到特殊 Token，找换行符 (bytes 10 -> \n)
        if best_offset == -1:
            newline_idx = mini_chunk.find(b"\n")
            if newline_idx != -1:
                # 在换行符之后切分
                best_offset = newline_idx + 1

        # 3. 如果都没找到，就强制在 target_position 切分
        if best_offset == -1:
            best_offset = 0

        final_pos = current_position + best_offset
        final_pos = min(final_pos, length)

        # 防止添加重复边界
        if final_pos > chunk_boundaries[-1]:
            chunk_boundaries.append(final_pos)

    if chunk_boundaries[-1] != length:
        chunk_boundaries.append(length)

    return sorted(list(set(chunk_boundaries)))


def _process_chunk_worker(args) -> Counter[bytes, int]:
    text, pattern_str, special_tokens = args

    token_counts = Counter()
    pat = re.compile(pattern_str)

    chunk_text = text.decode("utf-8", errors="ignore")

    if not chunk_text:
        return token_counts

    special_token_pattern = None
    special_tokens_set = set()
    if special_tokens:
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        escaped_tokens = [re.escape(tok) for tok in sorted_tokens]
        special_token_pattern = f"({'|'.join(escaped_tokens)})"
        parts = re.split(special_token_pattern, chunk_text)
        special_tokens_set = set(special_tokens)
    else:
        parts = [chunk_text]

    for part in parts:
        if not part:
            continue

        # 如果是特殊 Token，直接统计 bytes
        if part in special_tokens_set:
            token_counts[part.encode("utf-8")] += 1
            continue

        # 2. 对普通文本应用 GPT-2 正则分词
        for match in pat.finditer(part):
            token_bytes = match.group(0).encode("utf-8")
            token_counts[token_bytes] += 1

    return token_counts


def pretokenize_text(
    text: bytes,
    split_pattern: str = GPT2_SPLIT_PATTERN,
    special_tokens: list[str] | None = None,
) -> Counter:
    if special_tokens is None:
        special_tokens = []

    num_processes = max(1, CPU)

    boundaries = find_chunk_boundaries(text, num_processes, special_tokens)

    tasks = []
    for i in range(len(boundaries) - 1):
        tasks.append(
            (text[boundaries[i] : boundaries[i + 1]], split_pattern, special_tokens)
        )

    final_counts = Counter()

    if tasks:
        with multiprocessing.Pool(processes=min(len(tasks), num_processes)) as pool:
            for chunk_counter in pool.imap(_process_chunk_worker, tasks):
                final_counts.update(chunk_counter)

    return final_counts


def get_stats(
    pretokens_counts: dict[tuple[int, ...], int],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[tuple[int, ...]]]]:
    pairs_counts = defaultdict(int)
    pairs_words = defaultdict(set)

    for token_tuple, count in pretokens_counts.items():
        if len(token_tuple) < 2:
            continue
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pairs_counts[pair] += count
            pairs_words[pair].add(token_tuple)

    return pairs_counts, pairs_words


class HeapItem:
    def __init__(
        self, count: int, item0_bytes: bytes, item1_bytes: bytes, pair: tuple[int, int]
    ):
        self.count = count
        self.item0_bytes = item0_bytes
        self.item1_bytes = item1_bytes
        self.pair = pair

    def __lt__(self, other):
        # We want to pop the item with the LARGEST (count, item0_bytes, item1_bytes).
        # In a min-heap, the smallest item is popped first.
        # So we define 'lt' such that a larger item is considered 'smaller'.
        if self.count != other.count:
            return self.count > other.count
        if self.item0_bytes != other.item0_bytes:
            return self.item0_bytes > other.item0_bytes
        return self.item1_bytes > other.item1_bytes


def update_stats(
    pairs_counts: dict[tuple[int, int], int],
    pairs_words: dict[tuple[int, int], set[tuple[int, ...]]],
    best_pair: tuple[int, int],
    new_idx: int,
    token_counts: Counter,
) -> set[tuple[int, int]]:
    words_containing_pair = list(pairs_words[best_pair])
    p0, p1 = best_pair
    changes = set()

    for word in words_containing_pair:
        count = token_counts[word]

        # 生成新单词：贪婪匹配并替换 p0, p1
        new_word_list = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == p0 and word[i + 1] == p1:
                new_word_list.append(new_idx)
                i += 2
            else:
                new_word_list.append(word[i])
                i += 1

        new_word = tuple(new_word_list)

        # 1. 更新主词频
        token_counts[new_word] += count
        del token_counts[word]

        # 2. 处理旧单词中的 pair
        for i in range(len(word) - 1):
            old_pair = (word[i], word[i + 1])
            pairs_counts[old_pair] -= count
            changes.add(old_pair)

            if word in pairs_words[old_pair]:
                pairs_words[old_pair].remove(word)
                # 内存优化：如果集合空了，删除 key
                if not pairs_words[old_pair]:
                    del pairs_words[old_pair]

            if not pairs_counts[old_pair] and old_pair in pairs_counts:
                del pairs_counts[old_pair]

        # 3. 处理新单词中的 pair
        for i in range(len(new_word) - 1):
            new_pair = (new_word[i], new_word[i + 1])
            pairs_counts[new_pair] += count
            pairs_words[new_pair].add(new_word)
            changes.add(new_pair)

    # 清理已合并的 pair
    if best_pair in pairs_counts:
        del pairs_counts[best_pair]
    if best_pair in pairs_words:
        del pairs_words[best_pair]

    return changes


def test_run_train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        text = f.read()
    raw_token_counts = pretokenize_text(text, special_tokens=special_tokens)

    # 初始化词表 (0-255)
    vocab = {i: bytes([i]) for i in range(256)}

    # 映射特殊 Token
    special_token_map = {}
    base_special_idx = 256
    for i, st in enumerate(special_tokens):
        st_bytes = st.encode("utf-8")
        idx = base_special_idx + i
        special_token_map[st_bytes] = idx
        vocab[idx] = st_bytes

    # Counter[bytes] -> Counter[tuple[int]]
    token_counts = Counter()

    for token_bytes, count in raw_token_counts.items():
        if token_bytes in special_token_map:
            token_ids = (special_token_map[token_bytes],)
        else:
            token_ids = tuple(token_bytes)
        token_counts[token_ids] += count

    merges = []
    current_vocab_len = 256 + len(special_tokens)
    num_merges = vocab_size - current_vocab_len

    pairs_counts, pairs_words = get_stats(token_counts)

    hq = []
    for pair, count in pairs_counts.items():
        item = HeapItem(count, vocab[pair[0]], vocab[pair[1]], pair)
        heapq.heappush(hq, item)

    for i in range(num_merges):
        best_pair = None
        while hq:
            top_item = heapq.heappop(hq)
            if pairs_counts.get(top_item.pair, -1) == top_item.count:
                best_pair = top_item.pair
                break

        if best_pair is None:
            break

        new_idx = current_vocab_len + i
        part0_bytes = vocab[best_pair[0]]
        part1_bytes = vocab[best_pair[1]]

        merges.append((part0_bytes, part1_bytes))
        vocab[new_idx] = part0_bytes + part1_bytes

        changes = update_stats(
            pairs_counts, pairs_words, best_pair, new_idx, token_counts
        )

        for pair in changes:
            if pair in pairs_counts:
                count = pairs_counts[pair]
                item = HeapItem(count, vocab[pair[0]], vocab[pair[1]], pair)
                heapq.heappush(hq, item)

    return vocab, merges
