import setup
from setup import seed
import gym
import numpy as np
import torch
import wandb
import json
from args import args

import argparse
import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg, evaluate_episode_phi
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.encoder_transformer import EncoderTransformer
from decision_transformer.models.preference_decision_transformer import PreferenceDecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from decision_transformer.training.pdt_trainer import PDTTrainer
from reporter import get_reporter

from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE
import os
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    seed(variant['seed'])
    device = variant.get('device', 'cuda')

    env_name, dataset = variant['env'], variant['dataset']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000]
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    else:
        raise NotImplementedError
    env.seed(variant['seed'])

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dir_path = variant.get('dirpath', '.')
    dataset_path = f'{dir_path}/data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    pct_traj = variant.get('pct_traj', 1.)

    z_dim = variant['z_dim']
    print(f'z_dim is: {z_dim}')
    print(f"reward foresee is: {variant['foresee']}")

    expert_score = REF_MAX_SCORE[f"{variant['env']}-{variant['dataset']}-v2"]
    random_score = REF_MIN_SCORE[f"{variant['env']}-{variant['dataset']}-v2"]
    print(f"max score is: {expert_score}, min score is {random_score}")

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
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
    # TODO: 这里好像有问题，并不是在根据timesteps做sample
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    sub_trajectories = []

    SAMPLE_RATIO = 1 / 100 * 3
    def fill_batch(sub_trajectories, batch_size=256, max_len=K):
        while len(sub_trajectories) <= ((1e6 / K) * SAMPLE_RATIO):
            batch_inds = np.random.choice(
                np.arange(num_trajectories),
                size=batch_size,
                replace=True,
                p=p_sample,  # reweights so we sample according to timesteps
            )

            s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
            for i in range(batch_size):
                # TODO: 这里好像是优先级sample
                traj = trajectories[int(sorted_inds[batch_inds[i]])]
                si = random.randint(0, traj['rewards'].shape[0] - 1)

                # get sequences from dataset

                # reshape是在unsqueeze(0)
                # 如果si取到很后面的，si + max_len 就会超界，此时tlen < max_len
                s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
                a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
                r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
                if 'terminals' in traj:
                    d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
                else:
                    d.append(traj['dones'][si:si + max_len].reshape(1, -1))
                timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
                assert not (timesteps[-1] >= max_ep_len).any()
                timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff

                if variant['train_no_change']:
                    if not variant['subepisode']:
                        rtg.append(discount_cumsum(traj['rewards'][0:], gamma=1.)[0].reshape(1, 1, 1).repeat(s[-1].shape[1] + 1, axis=1))
                    else:
                        rtg.append(discount_cumsum(traj['rewards'][si:si+variant['foresee']], gamma=1.)[0].reshape(1, 1, 1).repeat(s[-1].shape[1] + 1, axis=1))
                        # rtg.append((np.sum(traj['rewards'][si:si + s[-1].shape[1]])).reshape(1,1,1).repeat(s[-1].shape[1] + 1, axis=1))

                else:
                    rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
                # print(f"rtg is: {rtg[-1]}")

                if rtg[-1].shape[1] <= s[-1].shape[1]:
                    assert False
                    rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

                # padding and state + reward normalization

                # TODO: 为什么在前面padding
                tlen = s[-1].shape[1]
                s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
                s[-1] = (s[-1] - state_mean) / state_std
                a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
                r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
                # TODO: 为什么要乘2
                d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
                rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
                timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
                mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
                sub_trajectories.append((s[-1].squeeze(0),a[-1].squeeze(0),rtg[-1][0, -1, 0], timesteps[-1].squeeze(0).astype(int), mask[-1].squeeze(0).astype(int)))

            # s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
            # a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
            # r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
            # d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
            # rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
            # timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
            # mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
            

    BACKET_NUM = 10
    fill_batch(sub_trajectories, max_len=variant['K'])
    sub_trajectories = sub_trajectories[:-(len(sub_trajectories) % BACKET_NUM)]
    print(f"get total {len(sub_trajectories)} sub_trajectories")

    sub_trajectories.sort(key=lambda se: se[2])


    en_model = EncoderTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=variant['embed_dim'],
        output_size=z_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=4*variant['embed_dim'],
        max_position_embeddings=1024,
        hidden_act=variant['activation_function'],
        hidden_dropout_prob=variant['dropout'],
        attention_probs_dropout_prob=variant['dropout'],
    )
    en_model = en_model.to(device=device)


    MODEL_WEIGHT = f"./tsne/model_weight/{variant['env']}_{variant['dataset']}.pt"
    (en_model_p, w, _) = torch.load(MODEL_WEIGHT)
    en_model.load_state_dict(en_model_p)
    en_model.eval()
    
    st_states = torch.stack([torch.from_numpy(state).float() for (state, _, _, _, _) in sub_trajectories]).to(device)
    st_actions = torch.stack([torch.from_numpy(action).float() for (_, action, _, _, _) in sub_trajectories]).to(device)
    st_timesteps = torch.stack([torch.from_numpy(t).long() for (_, _, _, t, _) in sub_trajectories]).to(device)
    st_masks = torch.stack([torch.from_numpy(mask).long() for (_, _, _, _, mask) in sub_trajectories]).to(device)

    en_model.eval()
    st_phis = en_model.forward(st_states, st_actions, st_timesteps, st_masks)
    tsne = TSNE(
        n_components=2,
        verbose=1,
        init="pca",
        method="exact",
        n_jobs=-1,
        random_state=variant['seed'],
    )
    z = tsne.fit_transform((torch.cat([st_phis, w.unsqueeze(0)], dim=0)).cpu().detach().numpy())
    z_min, z_max = z.min(0), z.max(0)
    z_norm = (z - z_min) / (z_max - z_min)

    z_norm = z_norm[:-1]
    w_z = z_norm[-1]
    
    df = pd.DataFrame()
    df["comp-1"] = z_norm[:, 0]
    df["comp-2"] = z_norm[:, 1]
    assert z_norm.shape[0] % BACKET_NUM == 0
    df["y"] = np.arange(z_norm.shape[0]) / z_norm.shape[0]

    df.to_csv(f"./tsne/dataframe/{variant['env']}_{variant['dataset']}.csv")
    np.save(f"./tsne/dataframe/{variant['env']}_{variant['dataset']}", w_z)
    # sns.scatterplot(
    #     x="comp-1",
    #     y="comp-2",
    #     hue=df.y.tolist(),
    #     # palette=sns.color_palette("hls", BACKET_NUM),
    #     data=df,
    # ).set(title="TSNE")
    # plt.scatter(x=w_z[0], y=w_z[1], marker="x", s=1000)
    # plt.savefig(f"./tsne/imgs/{variant['env']}_{variant['dataset']}.png")
    # plt.show()




if __name__ == '__main__':

    experiment('pbdt-tsne', variant=vars(args))
