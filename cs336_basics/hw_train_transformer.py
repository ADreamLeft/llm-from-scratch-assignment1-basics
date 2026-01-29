import math
from collections.abc import Callable, Iterable

import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    max_val = inputs.max(dim=-1, keepdim=True).values
    st = inputs - max_val
    exp_sum = torch.exp(st).sum(dim=-1)
    log_sum_exp = torch.log(exp_sum)

    target_logits = st.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    loss = log_sum_exp - target_logits

    return loss.mean()


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    if it > cosine_cycle_iters:
        return min_learning_rate
    cosine_decay = 0.5 * (
        1
        + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))
    )
    lr = min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_decay
    return lr


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return
    device = params[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in params]), 2
    )
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            p.grad.detach().mul_(clip_coef)


class SGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
            state = self.state[p]  # Get state associated with p.
            t = state.get(
                "t", 0
            )  # Get iteration number from the state, or initial value.
            grad = p.grad.data  # Get the gradient of loss with respect to p.
            p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
            state["t"] = t + 1  # Increment iteration number.

        return loss


class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "lam": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lam = group["lam"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["t"] = 0
                m = state["m"]
                v = state["v"]
                t = state["t"] + 1
                grad = p.grad.data

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * lam * p.data

                state["t"] = t
        return loss
