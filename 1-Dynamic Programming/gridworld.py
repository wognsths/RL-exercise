from typing import Set, Optional, Dict, Tuple, List
from mdp import MDP, State, Action, Transition

actions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
arrow4   = {0: "↑", 1: "→", 2: "↓", 3: "←"}

def build_gridworld(n_rows: int,
                    n_cols: int,
                    terminals: Dict[Tuple[int, int], float],
                    walls: Optional[Set[Tuple[int, int]]] = None,
                    step_cost: float = -0.01,
                    slip: float = 0.1,
                    gamma: float = 0.95) -> MDP:
    """
    Build an n_rows x n_cols gridworld with 4-way actions and stochastic slip.
    - terminals: dict {(r,c): reward} for absorbing states.
    - walls: set of blocked cells (no entry).
    - step_cost: reward for non-terminal transitions.
    - slip: with prob 'slip', action deviates to each perpendicular direction (split equally).
            intended dir prob = 1 - 2*slip.
    - hitting border/wall keeps you in place.
    """

    walls = walls or set()

    def to_id(r: int, c: int) -> int:
        return r * n_cols + c
    
    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < n_rows and 0 <= c < n_cols and (r, c) not in walls
    
    P: Dict[State, Dict[Action, List[Transition]]] = {}
    terminals_set = set(terminals.keys())

    for r in range(n_rows):
        for c in range(n_cols):
            s = to_id(r, c)
            if (r, c) in terminals_set:
                P[s] = {}
                continue
            P[s] = {}
            for a in range(4):
                trans: List[Transition] = []
                probs = []
                intended = a
                left = (a - 1) % 4
                right = (a + 1) % 4
                probs.append((intended, max(0.0, 1.0 - 2.0 * slip)))
                probs.append((left, slip))
                probs.append((right, slip))

                agg: Dict[Tuple[int, int], float] = {}
                for aa, pa in probs:
                    dr, dc = actions[aa]
                    nr, nc = r + dr, c + dc
                    if not in_bounds(nr, nc):
                        nr, nc = r, c
                    agg[(nr, nc)] = agg.get((nr, nc), 0.0) + pa

                for (nr, nc), p in agg.items():
                    sp = to_id(nr, nc)
                    if (nr, nc) in terminals_set:
                        reward = terminals[(nr, nc)]
                    else:
                        reward = step_cost
                    trans.append((p, sp, reward))
                
                P[s][a] = trans
    terminals_id = [to_id(r, c) for (r, c) in terminals_set]
    return MDP(P=P, nS=n_rows * n_cols, nA=4, gamma=gamma, terminals=terminals_id)

def pretty_value_grid(V: List[float], n_rows: int, n_cols: int) -> str:
    lines = []
    for r in range(n_rows):
        row = [f"{V[r*n_cols+c]:6.3f}" for c in range(n_cols)]
        lines.append(" ".join(row))
    return "\n".join(lines)
