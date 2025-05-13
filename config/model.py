from dataclasses import dataclass

@dataclass
class EnvConfig:
    episodes: int = 100_000


@dataclass
class PMCModelConfig:
    epochs: int = 10_000
    
    loss_fn: str = "CrossEntropyLoss"
    optim: str = "Adam"
    lr: float = 1e-4
