import cProfile
import json
import os
import pstats
import resource
import sys
import time

from cs336_basics.hw_train_bpe import test_run_train_bpe


def save_tokenizer(vocab, merges, vocab_path, merges_path):
    # Save vocab
    # vocab: dict[int, bytes] -> json with latin-1 strings
    vocab_str = {}
    for idx, token_bytes in vocab.items():
        vocab_str[str(idx)] = token_bytes.decode("latin-1")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, indent=2)

    # Save merges
    # merges: list[tuple[bytes, bytes]]
    merges_str = []
    for p1, p2 in merges:
        merges_str.append((p1.decode("latin-1"), p2.decode("latin-1")))

    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges_str, f)


def main():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    print(f"Starting BPE training on {input_path}...")
    start_time = time.time()

    # Profile the training
    profiler = cProfile.Profile()
    profiler.enable()

    vocab, merges = test_run_train_bpe(input_path, vocab_size, special_tokens)

    profiler.disable()
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # Memory usage
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # MacOS returns bytes, verify? No, getrusage usually returns kilobytes (on Linux) or bytes (on some systems).
    # On MacOS it seems to be in bytes. Wait, let's assume bytes and check if it's huge.
    # Actually on Mac it is bytes. On Linux it is kilobytes.
    # We can check sys.platform.
    if sys.platform == "darwin":
        memory_gb = usage / (1024 * 1024 * 1024)
    else:
        memory_gb = usage / (1024 * 1024)  # KB to GB

    print(f"Peak memory usage: {memory_gb:.2f} GB")

    # Save statistics
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(20)

    # Save results
    os.makedirs("output", exist_ok=True)
    save_tokenizer(vocab, merges, "output/vocab.json", "output/merges.txt")

    # Find longest token
    longest_token_bytes = b""
    for token in vocab.values():
        if len(token) > len(longest_token_bytes):
            longest_token_bytes = token

    print(f"Longest token length: {len(longest_token_bytes)}")
    try:
        print(
            f"Longest token (decoded): {longest_token_bytes.decode('utf-8', errors='replace')}"
        )
    except:
        print(f"Longest token (repr): {longest_token_bytes}")

    print(f"Vocabulary size: {len(vocab)}")


if __name__ == "__main__":
    main()
