import numpy as np
import torch
from tqdm import tqdm
from torch import nn

import time

from decision_transformer.training.trainer import Trainer


class PDTTrainer(Trainer):
    def __init__(
        self,
        en_model,
        de_model,
        optimizer,
        et_optimizer,
        w,
        w_optimizer,
        batch_size,
        get_batch,
        loss_fn,
        device,
        scheduler=None,
        eval_fns=None,
    ):
        super().__init__(None, optimizer, batch_size, get_batch, loss_fn, scheduler, eval_fns)
        self.en_model = en_model
        self.de_model = de_model
        self.et_optimizer = et_optimizer
        self.w = w
        self.w_optimizer = w_optimizer

        self.device = device

        self.regress_loss = nn.MSELoss()
        self.phi_loss = nn.MSELoss()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        # train_losses = []
        regress_losses = []
        recon_losses = []
        phi_norm_losses = []

        logs = dict()

        train_start = time.time()

        self.en_model.train()
        self.de_model.train()
        for i in tqdm(range(num_steps)):
            regress_loss, recon_loss, phi_norm_loss = self.train_step()
            regress_losses.append(regress_loss)
            recon_losses.append(recon_loss)
            phi_norm_losses.append(phi_norm_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.en_model.eval()
        self.de_model.eval()
        for eval_fn in self.eval_fns:
            # print(self.w)
            outputs = eval_fn((self.de_model,self.w.detach()))
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        # logs['training/train_loss_mean'] = np.mean(train_losses)
        # logs['training/train_loss_std'] = np.std(train_losses)
        logs['training/regress_loss_mean'] = np.mean(regress_losses)
        logs['training/regress_loss_std'] = np.std(regress_losses)
        logs['training/recon_loss_mean'] = np.mean(recon_losses)
        logs['training/recon_loss_std'] = np.std(recon_losses)
        logs['training/phi_norm_loss_mean'] = np.mean(phi_norm_losses)
        logs['training/phi_norm_loss_std'] = np.std(phi_norm_losses)
        print(self.w)
        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        # states_1, actions_1, rewards_1, dones_1, rtg_1, timesteps_1, attention_mask_1 = self.get_batch(self.batch_size)
        # states_2, actions_2, rewards_2, dones_2, rtg_2, timesteps_2, attention_mask_2 = self.get_batch(self.batch_size)

        # action_1_target = torch.clone(actions_1)
        # action_2_target = torch.clone(actions_2)

        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)

        action_target = torch.clone(actions)

        # TODO: 要传mask给et``
        _phi = self.en_model.forward(states, actions, timesteps, attention_mask)
        # if self.phi_norm == "hard":
        #     phi = _phi / torch.linalg.vector_norm(_phi.detach(), dim=1).unsqueeze(1)
        # else:

        phi = _phi
        phi_norm_loss = self.phi_loss(
                    torch.linalg.vector_norm(phi, dim=1),
                    torch.ones(self.batch_size).to(self.device),
                )

        pred_returns = torch.inner(phi, self.w.detach())
        # returns_loss = -(
        #     pred_returns
        #     / torch.linalg.vector_norm(self.w)
        #     / torch.linalg.vector_norm(phi)
        # ).mean()
        regress_loss = self.regress_loss(pred_returns, rtg[:,-1])

        phi = phi.unsqueeze(1).repeat_interleave(states.shape[1], dim=1)
        state_preds, action_preds, reward_preds = self.de_model.forward(
            states, actions, None, phi, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        recon_loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.et_optimizer.zero_grad()
        self.optimizer.zero_grad()
        (
            regress_loss
            + recon_loss
            # + 10 * returns_loss
            # + (phi_norm_loss if self.phi_norm == "soft" else 0)
            + phi_norm_loss
        ).backward()
        torch.nn.utils.clip_grad_norm_(self.en_model.parameters(), .25)
        torch.nn.utils.clip_grad_norm_(self.de_model.parameters(), .25)
        self.et_optimizer.step()
        self.optimizer.step()

        # # TODO: 也要传mask给et
        # phi = self.en_model.forward(states, actions, timesteps, attention_mask).detach()
        # pred_returns = torch.inner(phi, self.w)
        # regress_loss = self.regress_loss(pred_returns, rtg[:,-1])
        # self.w_optimizer.zero_grad()
        # regress_loss.backward()
        # self.w_optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return regress_loss.detach().cpu().item(), recon_loss.detach().cpu().item(), phi_norm_loss.detach().cpu().item()
