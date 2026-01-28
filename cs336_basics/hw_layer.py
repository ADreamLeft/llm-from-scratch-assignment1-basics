from math import sqrt

import torch
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn


def init_weights(module: nn.Module, std: float):
    if isinstance(module, Linear):
        torch.nn.init.trunc_normal_(
            module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    if isinstance(module, Embedding):
        torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3, b=3)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scores = torch.einsum("... q d, ... k d -> ... q k", Q, K) / sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = softmax(scores, dim=-1)
    output = torch.einsum("... q k, ... k v -> ... q v", attn_weights, V)

    return output


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = sqrt(2 / (in_features + out_features))
        init_weights(self, std=std)

    def forward(self, x: Float[Tensor, "... i"]) -> Float[Tensor, "... o"]:
        return torch.einsum("... i, o i -> ... o", x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        std = 1
        init_weights(self, std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RmsNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return norm_x * self.scale


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x3 = self.w3(x)
        return self.w2(self.silu(x1) * x3)


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k)
        )
        positions = torch.arange(max_seq_len, device=device)
        freqs = torch.einsum("i , j -> i j", positions, inv_freq)

        self.register_buffer("cos_emb", freqs.cos(), persistent=False)
        self.register_buffer("sin_emb", freqs.sin(), persistent=False)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> torch.Tensor:
        cos = self.cos_emb[token_positions]  # (..., seq_len, d_k // 2)
        sin = self.sin_emb[token_positions]  # (..., seq_len, d_k // 2)

        x_rope = x[..., : self.d_k]
        x_pass = x[..., self.d_k :]

        x1 = x_rope[..., 0::2]
        x2 = x_rope[..., 1::2]

        x_rope_out = torch.empty_like(x_rope)
        x_rope_out[..., 0::2] = x1 * cos - x2 * sin
        x_rope_out[..., 1::2] = x1 * sin + x2 * cos

        return torch.cat((x_rope_out, x_pass), dim=-1)


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RoPE | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = rope

        self.qkv_linear = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        q, k, v = rearrange(
            self.qkv_linear(x),
            "... s (three h d) -> three ...  h s d",
            three=3,
            h=self.num_heads,
            d=self.d_k,
        )

        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq_len = q.shape[-2]
        mask = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool).tril()

        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        output = rearrange(attn_out, "b h s d -> b s (h d)")

        return self.out_proj(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RoPE | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.attention = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            device=device,
            dtype=dtype,
        )
        self.rmsnorm1 = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.rmsnorm2 = RmsNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        batch_size, seq_len = x.shape[0], x.shape[1]
        token_positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )

        attn_out = self.attention(self.rmsnorm1(x), token_positions=token_positions)

        x = x + attn_out

        ffn_out = self.ffn(self.rmsnorm2(x))

        x = x + ffn_out

        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        rope: RoPE | None = None,
    ):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, context_length)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=context_length,
                    num_heads=8,
                    d_ff=4 * context_length,
                    rope=rope,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = RmsNorm(d_model=context_length)
        self.output_projection = Linear(context_length, vocab_size)

    def forward(
        self,
        token_ids: Int[Tensor, " batch_size sequence_length"],
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embedding(token_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.output_projection(x)

        return logits
