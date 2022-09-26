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

    df = pd.read_csv(f"./tsne/dataframe/{variant['env']}_{variant['dataset']}.csv")
    w_z = np.load(f"./tsne/dataframe/{variant['env']}_{variant['dataset']}.npy")
    sns.scatterplot(
        x="comp-1",
        y="comp-2",
        hue=df.y.tolist(),
        # palette=sns.color_palette("hls", BACKET_NUM),
        data=df,
    ).set(title="TSNE")
    plt.scatter(x=w_z[0], y=w_z[1], marker="x", s=1000)
    plt.savefig(f"./tsne/imgs/{variant['env']}_{variant['dataset']}.png")
    # plt.show()




if __name__ == '__main__':

    experiment('pbdt-tsne', variant=vars(args))
