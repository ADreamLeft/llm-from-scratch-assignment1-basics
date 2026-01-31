import cProfile
import json
import os
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


def compare_tokenizers(owt_vocab, ts_vocab_path):
    if not os.path.exists(ts_vocab_path):
        print(f"TinyStories vocab not found at {ts_vocab_path}, skipping comparison.")
        return

    print("\n--- Comparison with TinyStories Tokenizer ---")
    with open(ts_vocab_path, encoding="utf-8") as f:
        ts_vocab_raw = json.load(f)

    ts_vocab_set = set()
    for v in ts_vocab_raw.values():
        ts_vocab_set.add(v.encode("latin-1"))

    owt_vocab_set = set(owt_vocab.values())

    intersection = owt_vocab_set.intersection(ts_vocab_set)
    owt_only = owt_vocab_set - ts_vocab_set
    ts_only = ts_vocab_set - owt_vocab_set

    print(f"TinyStories Vocab Size: {len(ts_vocab_set)}")
    print(f"OpenWebText Vocab Size: {len(owt_vocab_set)}")
    print(f"Intersection Size: {len(intersection)}")
    print(f"Unique to OWT: {len(owt_only)}")
    print(f"Unique to TS: {len(ts_only)}")

    print("\nRandom samples unique to OWT:")
    import random

    if owt_only:
        for _ in range(min(10, len(owt_only))):
            t = random.choice(list(owt_only))
            try:
                print(f"  {t!r} -> {t.decode('utf-8')}")
            except:
                print(f"  {t!r}")

    print("\nLongest token in OWT:")
    longest_token_bytes = b""
    for token in owt_vocab_set:
        if len(token) > len(longest_token_bytes):
            longest_token_bytes = token
    print(f"Length: {len(longest_token_bytes)}")
    try:
        print(f"Decoded: {longest_token_bytes.decode('utf-8', errors='replace')}")
    except:
        print(f"Repr: {longest_token_bytes}")


def main():
    input_path = "data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    # Since the file is large (11GB), running full training might take hours.
    # The prompt implies I should do it ("Train a byte-level BPE tokenizer...").
    # But it also says "Resource requirements: <= 12 hours".

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
    if sys.platform == "darwin":
        memory_gb = usage / (1024 * 1024 * 1024)
    else:
        memory_gb = usage / (1024 * 1024)  # KB to GB

    print(f"Peak memory usage: {memory_gb:.2f} GB")

    # Save results
    os.makedirs("output", exist_ok=True)
    save_tokenizer(vocab, merges, "output/vocab_owt.json", "output/merges_owt.txt")

    compare_tokenizers(vocab, "output/vocab.json")


if __name__ == "__main__":
    main()
