import pickle

import numpy as np
import torch

import gym
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def experiment():
    device = "cpu" #"cuda"
    env = gym.make("Hopper-v3")
    max_ep_len = 1000
    env_targets = [3600, 1800]  # evaluation conditioning targets
    scale = 1000.0  # normalization for rewards/returns

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f"data/hopper-medium-v2.pkl"
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    observations, traj_lens, returns = [], [], []
    for path in trajectories:
        observations.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(path["rewards"].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    observations = np.concatenate(observations, axis=0)
    state_mean, state_std = np.mean(observations, axis=0), np.std(observations, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)
    K = 20
    batch_size = 64
    num_eval_episodes = 100
    pct_traj = 1.0

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        observations, actions, rewards, reward_to_go, timesteps, mask = [], [], [], [], [], []

        sorted_trajectories = [trajectories[sorted_inds[batch_idx]] for batch_idx in batch_inds]

        ep_lengths = np.array([trajectory["rewards"].shape[0] for trajectory in sorted_trajectories])
        start_idxs = np.random.randint(0, ep_lengths - max_len)
        # _observations = {
        #     key: np.array(
        #         [
        #             trajectory[key][start_idx : start_idx + max_len]
        #             for start_idx, trajectory in zip(start_idxs, sorted_trajectories)
        #         ],
        #         dtype=np.float32,
        #     )
        #     for key in sorted_trajectories[0].keys()
        # }

        # idxs = np.random.randint(0, ep_length - 1)
        # __observations = [trajectory["observations"][si : si + max_len].reshape(1, -1, obs_dim) for trajectory in sorted_trajectories]
        # alt_observations =

        for batch_idx, si in zip(batch_inds, start_idxs):
            trajectory = trajectories[sorted_inds[batch_idx]]

            # get sequences from dataset
            observations.append(trajectory["observations"][si : si + max_len].reshape(1, -1, obs_dim))
            actions.append(trajectory["actions"][si : si + max_len].reshape(1, -1, act_dim))
            rewards.append(trajectory["rewards"][si : si + max_len].reshape(1, -1, 1))
            timesteps.append(np.arange(si, si + observations[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            assert not (timesteps[-1] >= max_ep_len).any()
            reward_to_go.append(
                discount_cumsum(trajectory["rewards"][si:], gamma=1.0)[: observations[-1].shape[1] + 1].reshape(1, -1, 1)
            )
            if reward_to_go[-1].shape[1] <= observations[-1].shape[1]:
                reward_to_go[-1] = np.concatenate([reward_to_go[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = observations[-1].shape[1]
            observations[-1] = np.concatenate([np.zeros((1, max_len - tlen, obs_dim)), observations[-1]], axis=1)
            observations[-1] = (observations[-1] - state_mean) / state_std
            actions[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10.0, actions[-1]], axis=1)
            # rewards[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rewards[-1]], axis=1)
            reward_to_go[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), reward_to_go[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        observations = torch.from_numpy(np.concatenate(observations, axis=0)).to(dtype=torch.float32, device=device)
        actions = torch.from_numpy(np.concatenate(actions, axis=0)).to(dtype=torch.float32, device=device)
        rewards = torch.from_numpy(np.concatenate(rewards, axis=0)).to(dtype=torch.float32, device=device)
        reward_to_go = torch.from_numpy(np.concatenate(reward_to_go, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return observations, actions, rewards, reward_to_go, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        env,
                        obs_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew / scale,
                        mode="normal",
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                    )
                returns.append(ret)
                lengths.append(length)
            return {
                f"target_{target_rew}_return_mean": np.mean(returns),
                f"target_{target_rew}_return_std": np.std(returns),
                f"target_{target_rew}_length_mean": np.mean(lengths),
                f"target_{target_rew}_length_std": np.std(lengths),
            }

        return fn

    dropout = 0.1
    hidden_size = 128
    model = DecisionTransformer(
        state_dim=obs_dim,
        act_dim=act_dim,
        hidden_size=hidden_size,
        max_length=K,
        max_ep_len=max_ep_len,
        n_layer=3,
        n_head=1,
        n_inner=4 * hidden_size,
        activation_function="relu",
        n_positions=1024,
        resid_pdrop=dropout,
        attn_pdrop=dropout,
    )

    model = model.to(device=device)

    warmup_steps = 10_000
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=[eval_episodes(tar) for tar in env_targets],
    )
    for iter in range(10):
        trainer.train_iteration(num_steps=10_000, iter_num=iter + 1, print_logs=True)


if __name__ == "__main__":
    experiment()
