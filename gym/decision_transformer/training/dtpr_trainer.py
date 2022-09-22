import numpy as np
import torch
from tqdm import tqdm

import time

from decision_transformer.training.trainer import Trainer


class DTprTrainer(Trainer):

    def __init__(self, model, optimizer, reward_model, reward_optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None):
        super().__init__(model, optimizer, batch_size, get_batch, loss_fn, scheduler, eval_fns)
        self.reward_model = reward_model
        self.reward_optimizer = reward_optimizer
        self.CEloss = torch.nn.CrossEntropyLoss()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        reward_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for i in tqdm(range(num_steps)):
            train_loss, reward_loss = self.train_step()
            train_losses.append(train_loss)
            reward_losses.append(reward_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/reward_loss_mean'] = np.mean(reward_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        for _ in range(3):
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


        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size, reward_model = self.reward_model)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item(), pref_loss.detach().cpu().item()
