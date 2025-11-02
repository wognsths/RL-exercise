from typing import Optional, Set, Tuple, List

from mdp import *
from gridworld import *

def greedy_policy_arrows(mdp: MDP, V: List[float], n_rows: int, n_cols: int,
                         walls: Optional[Set[Tuple[int,int]]] = None,
                         terminals: Optional[Set[Tuple[int,int]]] = None) -> str:
    walls = walls or set()
    terminals = terminals or set()
    def to_id(r: int, c: int) -> int:
        return r * n_cols + c

    lines = []
    for r in range(n_rows):
        row_syms = []
        for c in range(n_cols):
            if (r, c) in walls:
                row_syms.append("■")
                continue
            if (r, c) in terminals:
                row_syms.append("T")
                continue
            s = to_id(r, c)
            a = greedy_action(mdp, V, s)
            row_syms.append(arrow4.get(a, "."))
        lines.append(" ".join(row_syms))
    return "\n".join(lines)

def policy_to_arrow_grid(pi: List[Optional[Action]], n_rows: int, n_cols: int,
                         walls: Optional[Set[Tuple[int, int]]] = None,
                         terminals: Optional[Set[Tuple[int, int]]] = None) -> str:
    walls = walls or set()
    terminals = terminals or set()

    def to_id(r: int, c: int) -> int:
        return r * n_cols + c

    lines = []
    for r in range(n_rows):
        row_syms = []
        for c in range(n_cols):
            if (r, c) in walls:
                row_syms.append("■")
                continue
            if (r, c) in terminals:
                row_syms.append("T")
                continue
            s = to_id(r, c)
            a = pi[s] if s < len(pi) else None
            row_syms.append(arrow4.get(a, ".") if a is not None else ".")
        lines.append(" ".join(row_syms))
    return "\n".join(lines)

if __name__ == "__main__":
    # Example: 4x4, terminals at (0,3)=+1, (1,3)=-1; one wall at (1,1)
    n_rows, n_cols = 4, 4
    terminals = {(0, 3): +1.0, (1, 3): -1.0}
    walls = {(1, 1)}

    gw = build_gridworld(n_rows, n_cols, terminals=terminals, walls=walls,
                         step_cost=-0.02, slip=0.1, gamma=0.95)

    print("== Gridworld: Value Iteration ==")
    vi_res = value_iteration(gw, tol=1e-10, trace=True)
    for step in vi_res.get("history", []):
        print(f"Iter {step['iteration']:3d} Δ={step['delta']:.3e}")
        print(pretty_value_grid(step["V"], n_rows, n_cols))
        print()
        time.sleep(1)
    Vg = vi_res["V"]
    print("Final greedy policy:")
    print(pretty_value_grid(Vg, n_rows, n_cols))
    print("Policy (greedy arrows):")
    print(greedy_policy_arrows(gw, Vg, n_rows, n_cols, walls=walls, terminals=set(terminals.keys())))

    print("== Gridworld: Policy Iteration ==")
    pi_res = policy_iteration(gw, trace=True, eval_trace=True)
    for step in pi_res.get("history", []):
        print(f"Outer loop {step['outer_loop']:2d} stable={step['stable']}")
        print("Value function:")
        print(pretty_value_grid(step["V"], n_rows, n_cols))
        print("Policy:")
        print(policy_to_arrow_grid(step["policy"], n_rows, n_cols, walls=walls, terminals=set(terminals.keys())))
        eval_hist = step.get("policy_evaluation", [])
        if eval_hist:
            print("  Policy evaluation deltas:")
            for eval_step in eval_hist:
                print(f"    iter {eval_step['iteration']:3d} Δ={eval_step['delta']:.3e}")
                time.sleep(0.1)
        print()
    Vg2 = pi_res["V"]
    print(pretty_value_grid(Vg2, n_rows, n_cols))
    print("Policy (greedy arrows):")
    print(greedy_policy_arrows(gw, Vg2, n_rows, n_cols, walls=walls, terminals=set(terminals.keys())))
