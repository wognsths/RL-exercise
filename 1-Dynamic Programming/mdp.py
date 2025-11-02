from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Any
import math
import time

State = int
Action = int
Transition = Tuple[float, State, float] # (prob, next_state, reward)


@dataclass
class MDP:
    P: Dict[State, Dict[Action, List[Transition]]] # Given state -> (prob, next_state, reward)
    nS: int
    nA: int
    gamma: float = 0.95
    terminals: List[State] = None

    def actions(self, s: State) -> List[Action]:
        if self.terminals and s in self.terminals:
            return []
        return list(self.P[s].keys())
    
def bellman_backup(mdp: MDP, V: List[float], s: State) -> float:
    """Compute (T* V)(s) = max_a sum_{s'} p(s'|s,a)[r + gamma V(s')]."""
    acts = mdp.actions(s)
    if not acts:
        return 0.0
    qs = []
    for a in acts:
        q = 0.0
        for p, sp, r in mdp.P[s][a]:
            q += p * (r + mdp.gamma * V[sp])
        qs.append(q)
    return max(qs)

def greedy_action(mdp: MDP, V: List[float], s: State) -> Action:
    best_a, best_q = None, -math.inf
    for a in mdp.actions(s):
        q = 0.0
        for p, sp, r in mdp.P[s][a]:
            q += p * (r + mdp.gamma * V[sp])
        if q > best_q:
            best_q, best_a = q, a
    return best_a

def value_iteration(
    mdp: MDP,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    trace: bool = False,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    V = [0.0] * mdp.nS
    t0 = time.time()
    history: Optional[List[Dict[str, Any]]] = [] if trace else None
    for it in range(1, max_iter + 1):
        delta = 0.0
        for s in range(mdp.nS):
            v_old = V[s]
            V[s] = bellman_backup(mdp, V, s)
            delta = max(delta, abs(v_old - V[s]))
        iteration_info = {"iteration": it, "delta": delta, "V": V.copy()}
        if callback:
            callback(iteration_info)
        if history is not None:
            history.append(iteration_info)
        if delta < tol:
            dt = time.time() - t0
            # Derive greedy policy from the converged V
            pi = [None] * mdp.nS
            for s in range(mdp.nS):
                pi[s] = greedy_action(mdp, V, s) if mdp.actions(s) else None
            result: Dict[str, Any] = {
                "V": V,
                "pi": pi,
                "iterations": it,
                "time_sec": dt,
                "converged": True,
            }
            if history is not None:
                result["history"] = history
            return result
    dt = time.time() - t0
    result = {
        "V": V,
        "pi": None,
        "iterations": max_iter,
        "time_sec": dt,
        "converged": False,
    }
    if history is not None:
        result["history"] = history
    return result

def policy_evaluation(
    mdp: MDP,
    pi: List[Action],
    tol: float = 1e-10,
    max_iter: int = 10_000,
    history: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[float]:
    V = [0.0] * mdp.nS
    for it in range(1, max_iter + 1):
        delta = 0.0
        for s in range(mdp.nS):
            if not mdp.actions(s):
                v_new = 0.0
            else:
                a = pi[s]
                v_new = 0.0
                for p, sp, r in mdp.P[s][a]:
                    v_new += p * (r + mdp.gamma * V[sp])
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        iteration_info = {"iteration": it, "delta": delta, "V": V.copy()}
        if callback:
            callback(iteration_info)
        if history is not None:
            history.append(iteration_info)
        if delta < tol:
            break
    return V


def policy_improvement(mdp: MDP, V: List[float], pi: List[Action]) -> Tuple[List[Action], bool]:
    policy_stable = True
    new_pi = pi.copy()
    for s in range(mdp.nS):
        if not mdp.actions(s):
            new_pi[s] = None
            continue
        old_a = pi[s]
        new_a = greedy_action(mdp, V, s)
        new_pi[s] = new_a
        if new_a != old_a:
            policy_stable = False
    return new_pi, policy_stable

def policy_iteration(
    mdp: MDP,
    tol_eval: float = 1e-10,
    max_eval_iter: int = 10_000,
    max_outer: int = 10_000,
    trace: bool = False,
    eval_trace: bool = False,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    eval_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    # Initialize with arbitrary valid actions per state
    pi = [None if not mdp.actions(s) else mdp.actions(s)[0] for s in range(mdp.nS)]
    t0 = time.time()
    history: Optional[List[Dict[str, Any]]] = [] if trace else None
    for k in range(1, max_outer + 1):
        prev_pi = pi.copy()
        eval_history: Optional[List[Dict[str, Any]]] = [] if eval_trace else None
        V = policy_evaluation(
            mdp,
            pi,
            tol=tol_eval,
            max_iter=max_eval_iter,
            history=eval_history,
            callback=eval_callback,
        )
        new_pi, stable = policy_improvement(mdp, V, pi)
        pi = new_pi
        iteration_info = {
            "outer_loop": k,
            "V": V.copy(),
            "policy": pi.copy(),
            "previous_policy": prev_pi,
            "stable": stable,
        }
        if eval_history is not None:
            iteration_info["policy_evaluation"] = eval_history
        if callback:
            callback(iteration_info)
        if history is not None:
            history.append(iteration_info)
        if stable:
            dt = time.time() - t0
            result: Dict[str, Any] = {
                "V": V,
                "pi": pi,
                "outer_loops": k,
                "time_sec": dt,
                "converged": True,
            }
            if history is not None:
                result["history"] = history
            return result
    dt = time.time() - t0
    result = {
        "V": V,
        "pi": pi,
        "outer_loops": max_outer,
        "time_sec": dt,
        "converged": False,
    }
    if history is not None:
        result["history"] = history
    return result
