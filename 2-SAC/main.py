import argparse
import csv
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from config import Args
from replay_memory import ReplayMemory
from sac import SAC


def parse_args():
    default_args = Args()
    parser = argparse.ArgumentParser(description="Soft Actor-Critic (v1) trainer")

    # Environment / training loop
    parser.add_argument("--env-name", type=str, default="Pendulum-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--memory-size", type=int, default=1_000_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--deterministic-eval",
        dest="deterministic_eval",
        action="store_true",
        help="Use mean action during evaluation",
    )
    parser.add_argument(
        "--stochastic-eval",
        dest="deterministic_eval",
        action="store_false",
        help="Use sampled action during evaluation",
    )
    parser.set_defaults(deterministic_eval=True)

    # SAC hyperparameters
    parser.add_argument("--gamma", type=float, default=default_args.gamma)
    parser.add_argument("--tau", type=float, default=default_args.tau)
    parser.add_argument("--alpha", type=float, default=default_args.alpha)
    parser.add_argument("--hidden-dim", type=int, default=default_args.hidden_dim)
    parser.add_argument("--n-q", type=int, default=default_args.n_QNetwork)
    parser.add_argument("--lr", type=float, default=default_args.lr)

    return parser.parse_args()


def set_seed_everywhere(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_env(env, seed=None):
    reset_out = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(reset_out, tuple):
        return reset_out[0]
    return reset_out


def step_env(env, action):
    step_out = env.step(action)
    if len(step_out) == 5:
        next_state, reward, terminated, truncated, _ = step_out
        done = terminated or truncated
    else:
        next_state, reward, done, _ = step_out
    return next_state, reward, done


def evaluate_policy(env_name, seed, agent, episodes, deterministic):
    eval_env = gym.make(env_name)
    if hasattr(eval_env.action_space, "seed"):
        eval_env.action_space.seed(seed + 10_000)
    if hasattr(eval_env.observation_space, "seed"):
        eval_env.observation_space.seed(seed + 10_000)

    returns = []
    for idx in range(episodes):
        state = np.asarray(reset_env(eval_env, seed + idx), dtype=np.float32).reshape(
            -1
        )
        done = False
        episode_return = 0.0

        while not done:
            action = agent.select_action(state, evaluate=deterministic)
            next_state, reward, done = step_env(eval_env, action)
            state = np.asarray(next_state, dtype=np.float32).reshape(-1)
            episode_return += reward

        returns.append(episode_return)

    eval_env.close()
    return float(np.mean(returns)), float(np.std(returns))


def save_checkpoint(agent, checkpoint_dir, env_name, seed, step, memory):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"sac_{env_name}_seed{seed}_step{step}.pt"
    torch.save(
        {
            "step": step,
            "policy": agent.policy.state_dict(),
            "value": agent.value.state_dict(),
            "target_value": agent.target_value.state_dict(),
            "critic": agent.critic.state_dict(),
            "policy_optim": agent.policy_optim.state_dict(),
            "value_optim": agent.value_optim.state_dict(),
            "critic_optim": agent.critic_optim.state_dict(),
        },
        checkpoint_path,
    )

    buffer_path = checkpoint_dir / f"sac_buffer_{env_name}_seed{seed}_step{step}.pkl"
    memory.save_buffer(env_name, suffix=str(step), save_path=str(buffer_path))


def main():
    args = parse_args()
    set_seed_everywhere(args.seed)

    env = gym.make(args.env_name)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(args.seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(args.seed)

    state = np.asarray(reset_env(env, args.seed), dtype=np.float32).reshape(-1)
    obs_dim = int(np.prod(state.shape))

    memory = ReplayMemory(args.memory_size, args.seed)
    sac_args = Args(
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        hidden_dim=args.hidden_dim,
        n_QNetwork=args.n_q,
        lr=args.lr,
    )
    agent = SAC(obs_dim, env.action_space, sac_args, memory)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.env_name}_seed{args.seed}.csv"
    log_file = log_path.open("w", newline="")
    writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "step",
            "episode",
            "episode_length",
            "episode_return",
            "value_loss",
            "q_loss",
            "policy_loss",
            "alpha",
            "eval_return",
            "eval_std",
        ],
    )
    writer.writeheader()
    log_file.flush()

    total_steps = 0
    episode = 1
    episode_reward = 0.0
    episode_length = 0
    last_losses = {"value_loss": None, "q_loss": None, "policy_loss": None}

    try:
        while total_steps < args.total_steps:
            if total_steps < args.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, done = step_env(env, action)
            next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
            memory.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            if len(memory) > args.batch_size and total_steps >= args.start_steps:
                for _ in range(args.updates_per_step):
                    value_loss, q_loss, policy_loss = agent.update_params(
                        args.batch_size, total_steps
                    )
                    last_losses = {
                        "value_loss": value_loss,
                        "q_loss": q_loss,
                        "policy_loss": policy_loss,
                    }

            if done:
                writer.writerow(
                    {
                        "step": total_steps,
                        "episode": episode,
                        "episode_length": episode_length,
                        "episode_return": episode_reward,
                        "value_loss": last_losses["value_loss"],
                        "q_loss": last_losses["q_loss"],
                        "policy_loss": last_losses["policy_loss"],
                        "alpha": sac_args.alpha,
                        "eval_return": None,
                        "eval_std": None,
                    }
                )
                log_file.flush()

                if episode % args.log_interval == 0:
                    print(
                        f"[train] step {total_steps} | episode {episode} | "
                        f"return {episode_reward:.2f} | length {episode_length}"
                    )

                state = np.asarray(reset_env(env), dtype=np.float32).reshape(-1)
                episode_reward = 0.0
                episode_length = 0
                episode += 1

            if args.eval_interval > 0 and total_steps % args.eval_interval == 0:
                eval_mean, eval_std = evaluate_policy(
                    args.env_name,
                    args.seed,
                    agent,
                    args.eval_episodes,
                    args.deterministic_eval,
                )
                print(
                    f"[eval] step {total_steps} | return {eval_mean:.2f} +/- {eval_std:.2f}"
                )
                writer.writerow(
                    {
                        "step": total_steps,
                        "episode": None,
                        "episode_length": None,
                        "episode_return": None,
                        "value_loss": last_losses["value_loss"],
                        "q_loss": last_losses["q_loss"],
                        "policy_loss": last_losses["policy_loss"],
                        "alpha": sac_args.alpha,
                        "eval_return": eval_mean,
                        "eval_std": eval_std,
                    }
                )
                log_file.flush()
                save_checkpoint(
                    agent,
                    args.checkpoint_dir,
                    args.env_name,
                    args.seed,
                    total_steps,
                    memory,
                )
            elif (
                args.checkpoint_interval > 0
                and total_steps % args.checkpoint_interval == 0
            ):
                save_checkpoint(
                    agent,
                    args.checkpoint_dir,
                    args.env_name,
                    args.seed,
                    total_steps,
                    memory,
                )

    finally:
        env.close()
        log_file.close()


if __name__ == "__main__":
    main()
