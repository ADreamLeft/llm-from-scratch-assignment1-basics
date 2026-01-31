import json
import os
import time
from typing import Any

import numpy as np
import torch

from cs336_basics.hw_layer import RoPE, TransformerLM
from cs336_basics.hw_train_transformer import (
    AdamWOptimizer,
    cross_entropy,
    gradient_clipping,
    lr_cosine_schedule,
)
from cs336_basics.hw_training_loop import data_loading, evaluate_loss, save_checkpoint

# Check device
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using CUDA")
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using MPS")
else:
    DEVICE = "cpu"
    print("Using CPU")


class ExperimentLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.logs = []
        # Clear/Create file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            pass

    def log(self, metrics: dict[str, Any]):
        metrics["timestamp"] = time.time()
        self.logs.append(metrics)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Simple print for progress
        if "step" in metrics and "loss" in metrics:
            print(
                f"Step {metrics['step']}: Loss {metrics['loss']:.4f} (Val: {metrics.get('val_loss', 'N/A')})"
            )


def train_model(
    config: dict[str, Any],
    train_data: np.ndarray,
    val_data: np.ndarray,
    experiment_name: str,
    output_dir: str = "experiments",
):
    print(f"\nStarting Experiment: {experiment_name}")
    print(f"Config: {config}")

    # Unpack config
    batch_size = config["batch_size"]
    context_length = config["context_length"]
    d_model = config["d_model"]
    d_ff = config["d_ff"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    vocab_size = config["vocab_size"]
    total_tokens = config["total_tokens"]
    learning_rate = config["learning_rate"]
    warmup_iters = config["warmup_iters"]
    min_learning_rate = config["min_learning_rate"]

    # Derived config
    tokens_per_step = batch_size * context_length
    total_steps = total_tokens // tokens_per_step

    print(f"Total Steps: {total_steps}")

    # Check if we need RoPE instantiation.
    # The instructions say "RoPE theta parameter 10000".
    # `TransformerLM` passes `rope` to `TransformerBlock` -> `MultiheadSelfAttention`.
    # `MultiheadSelfAttention` uses `rope` if valid.
    # So we should probably instantiate RoPE and pass it.

    rope = RoPE(
        theta=10000.0,
        d_k=d_model // num_heads,
        max_seq_len=context_length,
        device=DEVICE,
    )

    # Re-init model with RoPE + Device
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope=rope,
        device=DEVICE,
    )
    # Move model to device (init does recursive parameter creation on device if specified, but better be safe)
    model.to(DEVICE)

    if DEVICE == "cpu":
        model = torch.compile(model)
    elif DEVICE == "mps":
        model = torch.compile(model, backend="aot_eager")

    # Optimizer
    optimizer = AdamWOptimizer(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,  # Default mentioned in HW text or Kingma Ba
    )

    # Logger
    logger = ExperimentLogger(os.path.join(output_dir, f"{experiment_name}.jsonl"))

    start_time = time.time()

    for step in range(1, total_steps + 1):
        # LR Schedule
        lr = lr_cosine_schedule(
            step, learning_rate, min_learning_rate, warmup_iters, total_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Data
        inputs, labels = data_loading(train_data, batch_size, context_length, DEVICE)

        # Forward
        optimizer.zero_grad()
        logits = model(inputs)

        # Loss
        B, T, V = logits.shape
        loss = cross_entropy(logits.reshape(B * T, V), labels.reshape(B * T))

        # Check divergence
        if torch.isnan(loss) or loss.item() > 20.0:
            print(f"Diverged at step {step} with loss {loss.item()}")
            logger.log({"step": step, "loss": float(loss.item()), "status": "diverged"})
            return float("inf")

        # Backward
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)
        optimizer.step()

        metrics = {
            "step": step,
            "loss": loss.item(),
            "lr": lr,
            "time_elapsed": time.time() - start_time,
        }
        if step % 10 == 0 or step == total_steps:
            val_loss = evaluate_loss(
                model, val_data, batch_size, context_length, DEVICE, max_batches=10
            )
            metrics["val_loss"] = val_loss

            # Checkpoint
            save_checkpoint(
                model,
                optimizer,
                step,
                os.path.join(output_dir, f"{experiment_name}_ckpt.pt"),
            )
        # Logging & Valid
        if step % 50 == 0 or step == total_steps:
            logger.log(metrics)

    final_val_loss = evaluate_loss(
        model, val_data, batch_size, context_length, DEVICE, max_batches=20
    )
    print(f"Final Validation Loss: {final_val_loss}")
    return final_val_loss


def train_worker(args):
    config, train_path, valid_path, exp_name = args
    # Re-load or mmap data in worker to avoid serialization cost
    train_data = np.load(train_path, mmap_mode="r")
    valid_data = np.load(valid_path, mmap_mode="r")
    return train_model(config, train_data, valid_data, exp_name)


def main():
    # Load Data
    print("Loading Data...")
    train_path = "data/tinystories_train_ids.npy"
    valid_path = "data/tinystories_valid_ids.npy"

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run tokenizer_experiments.py first.")
        return

    # Use mmap_mode='r' to save memory and allow pickling arrays if needed,
    # though we pass paths to workers.
    train_ids = np.load(train_path, mmap_mode="r")
    valid_ids = np.load(valid_path, mmap_mode="r")

    # 7.2 & 7.3 Base Configuration
    # Low resource adjustment
    # REAL ASSIGNMENT VALUE: 40_000_000 (Low Resource) or 327_680_000 (Full)
    # FOR DEMO/TESTING: Using small value to verify pipeline works in reasonable time.
    final_tokens = 40_000_000

    base_config = {
        "batch_size": 64,
        "context_length": 256,
        "d_model": 512,
        "d_ff": 1344,
        "num_layers": 4,
        "num_heads": 16,
        "vocab_size": 10000,
        "total_tokens": final_tokens,
        "learning_rate": 5e-4,  # Initial guess
        "min_learning_rate": 5e-5,
        "warmup_iters": 100,
    }

    best_lr = 1e-3
    # --- Problem (batch_size_experiment) ---
    print("\n--- Problem: Batch Size Variations ---")
    batch_sizes = [192, 128]
    bs_losses = {}
    for bs in batch_sizes:
        if bs == 64:
            continue  # Already run in LR sweep (assuming 32 was used)
        cfg = base_config.copy()
        cfg["learning_rate"] = best_lr
        cfg["batch_size"] = bs
        loss = train_model(cfg, train_ids, valid_ids, f"bs_sweep_{bs}")
        bs_losses[bs] = loss

    best_bs = min(bs_losses, key=bs_losses.get)

    # --- Problem (generate) ---
    print("\n--- Problem: Generate Text ---")
    # Load best model
    best_ckpt = f"experiments/bs_sweep_{best_bs}_ckpt.pt"
    if not os.path.exists(best_ckpt):
        print("Best checkpoint not found.")
        return

    # Text Generation
    # Need to load model architecture again
    model = TransformerLM(
        vocab_size=base_config["vocab_size"],
        d_model=base_config["d_model"],
        num_layers=base_config["num_layers"],
        num_heads=base_config["num_heads"],
        d_ff=base_config["d_ff"],
        rope=RoPE(
            theta=10000.0,
            d_k=base_config["d_model"] // base_config["num_heads"],
            max_seq_len=base_config["context_length"],
            device=DEVICE,
        ),
        device=DEVICE,
    )
    checkpoint = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # Decode
    from cs336_basics.hw_tokenizer import TestTokenizer

    tokenizer = TestTokenizer.from_files(
        "output/vocab.json", "output/merges.txt", ["<|endoftext|>"]
    )

    prompt = "Once upon a time, there was a"
    prompt_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    print(f"Prompt: {prompt}")
    generated_ids = model.generate(
        input_tensor, max_new_tokens=200, temperature=0.8, top_p=0.9
    )

    # Validating ID range before decode
    out_ids = generated_ids[0].tolist()
    vocab_map = tokenizer.vocab
    # vocab_map is id -> bytes
    decoded_bytes = b""
    for i in out_ids:
        if i in vocab_map:
            decoded_bytes += vocab_map[i]

    decoded_text = decoded_bytes.decode("utf-8", errors="replace")
    print(f"Generated:\n{decoded_text}")


if __name__ == "__main__":
    from cs336_basics.hw_layer import RoPE  # Import inside or top level

    main()
