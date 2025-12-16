import time
import numpy as np
import torch
import gymnasium as gym

from models import GaussianPolicy

DEVICE = "cpu"
ENV_NAME = "HalfCheetah-v4"
PT_PATH  = "checkpoints/sac_HalfCheetah-v4_seed0_step1000000.pt"
HIDDEN_DIM = 256

def main():
    env = gym.make(ENV_NAME, render_mode = "human")
    obs, _ = env.reset(seed=0)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = GaussianPolicy(obs_dim, act_dim, HIDDEN_DIM, action_space=env.action_space).to(DEVICE)
    ckpt = torch.load(PT_PATH, map_location=DEVICE)
    policy.load_state_dict(ckpt["policy"])

    policy.eval()

    while True:
        state = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = policy.sample(state, eval=True)

        action = action.squeeze(0).cpu().numpy()

        obs, r, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

        time.sleep(1/60)


if __name__=="__main__":
    main()