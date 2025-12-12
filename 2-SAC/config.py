from dataclasses import dataclass

@dataclass
class Args:
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    hidden_dim: int = 256
    n_QNetwork: int = 2
    lr: float = 1e-5

@dataclass
class Env:
    env: str
    seed: int
