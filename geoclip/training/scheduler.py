import math
import torch.optim as optim


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> optim.lr_scheduler.LambdaLR:
    """
    Cosine annealing LR schedule with a linear warmup phase.
    LR rises linearly from 0 → base_lr over warmup_steps, then follows a
    cosine curve down to min_lr_ratio × base_lr by the end of training.
    """

    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # Cosine decay
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
