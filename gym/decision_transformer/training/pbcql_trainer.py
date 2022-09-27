import numpy as np
import torch
from tqdm import tqdm

import time

from decision_transformer.training.trainer import Trainer
import gym
import d4rl
import d3rlpy
from d3rlpy.dataset import MDPDataset
import math


class CQLTrainer(Trainer):

    def __init__(self, model, batch_size, reward_model, reward_optimizer,  get_batch, dataset, state_mean, state_std, device, model_type,  scheduler=None, eval_fns=None):
        super().__init__(model, None, batch_size, get_batch, None, scheduler, eval_fns)
        self.reward_model = reward_model
        self.reward_optimizer = reward_optimizer
        self.CEloss = torch.nn.CrossEntropyLoss()

        self.trained_iterations = 0
        self.dataset = dataset
        self.state_mean=torch.from_numpy(state_mean).float().to(device)
        self.state_std=torch.from_numpy(state_std).float().to(device)

        self.o_tensor = (torch.from_numpy(self.dataset['observations']).float().to(device) - self.state_mean) / self.state_std
        self.a_tensor = torch.from_numpy(self.dataset['actions']).float().to(device)
        self.model_type = model_type
        assert self.model_type in ['ocql', 'pbcql']
    
    def train_rewarder(self, num_iters = int(1e4), reporter = None):
        for _ in range(num_iters):
            # Review: 这里没来得及看，可能需要仔细对下看和PbDT是否一致
            states_1, actions_1, rewards_1, dones_1, rtg_1, timesteps_1, attention_mask_1 = self.get_batch(self.batch_size, rew_tra = True)
            states_2, actions_2, rewards_2, dones_2, rtg_2, timesteps_2, attention_mask_2 = self.get_batch(self.batch_size, rew_tra = True)
            rtg_hat_1 = self.reward_model(states_1, actions_1)
            rtg_hat_2 = self.reward_model(states_2, actions_2)
            rtg_hat_1 = torch.mul(rtg_hat_1,attention_mask_1).sum(-1).unsqueeze(-1)
            rtg_hat_2 = torch.mul(rtg_hat_2,attention_mask_2).sum(-1).unsqueeze(-1)

            # rtg_hat_1 = rtg_hat_1.reshape(-1, 1)[attention_mask_1.reshape(-1) > 0].reshape(self.batch_size, -1).sum(-1)
            # rtg_hat_2 = rtg_hat_2.reshape(-1, 1)[attention_mask_2.reshape(-1) > 0].reshape(-1, 20).sum(-1)
            # rtg_hat_2 = torch.mul(rtg_hat_2,attention_mask_2).sum(-1)
            pref_hat = torch.cat([rtg_hat_1, rtg_hat_2], axis=-1)

            pref_1 = (rtg_1[:,-1,0]>rtg_2[:,-1,0]).to(dtype=torch.float32).unsqueeze(-1)
            pref_2 = (rtg_2[:,-1,0]>rtg_1[:,-1,0]).to(dtype=torch.float32).unsqueeze(-1)
            pref = torch.cat([pref_1, pref_2], axis=-1)
            pref_loss = self.CEloss(pref_hat, pref.detach())

            self.reward_optimizer.zero_grad()
            pref_loss.backward()
            self.reward_optimizer.step()

            if reporter is not None:
                reporter(dict(pref_loss=pref_loss.item()))
            
    def relabel(self):

        rewards = np.empty((self.o_tensor.size(0), ))

        # 是为了切分再在gpu上预测，不然可能cuda oom
        # Review: 可能需要仔细看下
        for i in range(math.ceil(self.o_tensor.size(0) / 100)):
            rewards[(i*100):(i*100+100)] = self.reward_model(self.o_tensor[(i*100):(i*100+100)], self.a_tensor[(i*100):(i*100+100)]).detach().cpu().numpy()

        terminals = self.dataset["terminals"]
        timeouts = self.dataset["timeouts"]
        episode_terminals = np.logical_or(terminals, timeouts)

        self.mdp_dataset = MDPDataset(
            # observations=np.array(observations, dtype=np.float32),
            # actions=np.array(actions, dtype=np.float32),
            observations=self.o_tensor.cpu().detach().numpy(),
            actions=self.a_tensor.cpu().detach().numpy(),
            rewards=np.array(self.dataset['rewards'] if self.model_type == 'ocql' else rewards, dtype=np.float32),
            terminals=np.array(terminals, dtype=np.float32),
            episode_terminals=np.array(episode_terminals, dtype=np.float32),
        )


    def train_iteration(self, num_steps, iter_num=0, print_logs=False, reporter = None):

        logs = dict()

        train_start = time.time()


            # if self.scheduler is not None:
            #     self.scheduler.step()

        # self.model.fit(self.mdp_dataset, n_steps=num_steps, n_steps_per_epoch=num_steps, save_metrics=False, verbose=False)

        assert num_steps % 10 == 0
        # d3rlpy提供的训练接口，训1000个epoch，每个epoch 10次训练，总共100个epoch（为了方便统计）
        info = self.model.fit(self.mdp_dataset, n_steps=int(num_steps * 4.5), n_steps_per_epoch=int(num_steps * 4.5) / 100, save_metrics=False,verbose=False, show_progress=False)

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        # self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        # logs['training/alpha'] = np.mean([i[1]['alpha'] for i in info])
        # logs['training/alpha_loss_mean'] = np.mean([i[1]['alpha_loss'] for i in info])
        logs['training/actor_loss_mean'] = np.mean([i[1]['actor_loss'] for i in info])
        logs['training/critic_loss_mean'] = np.mean([i[1]['critic_loss'] for i in info])
        # logs['training/train_loss_mean'] = np.mean(train_losses)
        # logs['training/train_loss_std'] = np.std(train_losses)
        # logs['training/reward_loss_mean'] = np.mean(reward_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        self.trained_iterations += 1
        return logs

    def train_step(self, reporter):
        # interleave的reward训练
        self.train_rewarder(1, reporter)
