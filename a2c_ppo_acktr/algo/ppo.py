import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from opacus import PrivacyEngine
# from opacus.utils.uniform_sampler import UniformWithReplacementSampler
# from grad_tools.pcgrad import PCGrad
# from grad_tools.noisygrad import NoisyGrad
# from grad_tools.testgrad import TestGrad
# from grad_tools.graddrop import GradDrop
# from grad_tools.mediangrad import MedianGrad
# from grad_tools.meanvar_grad import MeanVarGrad
from grad_tools.grad_plot import GradPlot


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 l2_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=False,
                 use_privacy=False,
                 use_pcgrad=False,
                 use_testgrad=False,
                 use_testgrad_median=False,
                 use_median_grad=False,
                 testgrad_quantile=-1,
                 use_noisygrad=False,
                 use_meanvargrad=False,
                 meanvar_beta=1.0,
                 use_graddrop=False,
                 no_special_grad_for_critic=False,
                 attention_policy=False,
                 max_task_grad_norm=1.0,
                 testgrad_alpha=1.0,
                 testgrad_beta=1.0,
                 grad_noise_ratio=1.0,
                 num_tasks=0,
                 weight_decay=0.0):

        self.actor_critic = actor_critic
        self.num_tasks = num_tasks
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.l2_coef = l2_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.attention_parameters = []
        self.non_attention_parameters = []
        for name, p in actor_critic.named_parameters():
            if 'attention' in name:
                self.attention_parameters.append(p)
            else:
                self.non_attention_parameters.append(p)

        # no_special_grad_for_critic means that we apply a standard gradient to the critic parameters and a special
        # gradient (e.g., testgrad) to the actor parameters. To do that, we name the different parameter groups in the
        # optimizer, and modify the special gradient code to take that into account
        if no_special_grad_for_critic:
            critic_params = []
            non_critic_params = []
            for name, p in actor_critic.named_parameters():
                if 'critic' in name:
                    critic_params.append(p)
                else:
                    non_critic_params.append(p)
            self.optimizer = optim.Adam([{'params': critic_params,
                                          'special_grad': False},
                                        {'params': non_critic_params,
                                         'special_grad': True}],
                                        lr=lr, eps=eps, weight_decay=weight_decay)
        else:
            if attention_policy:
                self.optimizer = optim.Adam(self.attention_parameters, lr=lr, eps=eps, weight_decay=weight_decay)
            else:
                self.optimizer = optim.Adam(self.non_attention_parameters, lr=lr, eps=eps, weight_decay=weight_decay)
        self.attention_policy = attention_policy
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=4000, gamma=0.1)
        # self.optimizer = GradPlot(self.optimizer)
        # self.max_task_grad_norm = max_task_grad_norm
        # self.use_pcgrad = use_pcgrad
        # self.use_testgrad = use_testgrad
        # self.use_noisygrad = use_noisygrad
        # self.use_median_grad = use_median_grad
        # self.use_meanvargrad = use_meanvargrad
        # self.use_graddrop = use_graddrop
        # self.use_privacy = use_privacy
        # if use_pcgrad:
        #     self.optimizer = PCGrad(self.optimizer)
        # if use_graddrop:
        #     self.optimizer = GradDrop(self.optimizer)
        # if use_testgrad:
        #     if use_testgrad_median:
        #         self.optimizer = TestGrad(self.optimizer,
        #                                   use_median=True,
        #                                   max_grad_norm=num_mini_batch * max_task_grad_norm,
        #                                   noise_ratio=grad_noise_ratio)
        #     else:
        #         self.optimizer = TestGrad(self.optimizer,
        #                                   use_median=False,
        #                                   max_grad_norm=num_mini_batch * max_task_grad_norm,
        #                                   noise_ratio=grad_noise_ratio,
        #                                   quantile=testgrad_quantile,
        #                                   alpha=testgrad_alpha,
        #                                   beta=testgrad_beta)
        # if use_noisygrad:
        #     self.optimizer = NoisyGrad(self.optimizer,
        #                                max_grad_norm=num_mini_batch * max_task_grad_norm,
        #                                noise_ratio=grad_noise_ratio)
        # if use_meanvargrad:
        #     self.optimizer = MeanVarGrad(self.optimizer,
        #                                  max_grad_norm=num_mini_batch * max_task_grad_norm,
        #                                  noise_ratio=grad_noise_ratio,
        #                                  beta=meanvar_beta)
        # if use_median_grad:
        #     self.optimizer = MedianGrad(self.optimizer,
        #                                 noise_ratio=grad_noise_ratio)
        # if use_privacy:
        #     privacy_engine = PrivacyEngine(
        #         actor_critic,
        #         sample_rate=0.01,
        #         noise_multiplier=grad_noise_ratio,
        #         max_grad_norm=max_task_grad_norm,
        #     )
        #     privacy_engine.attach(self.optimizer)

    def update(self, rollouts, attention_update=False, make_update=True):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        dist_l2_epoch = 0
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                # data_generators = [rollouts.recurrent_generator(
                #     advantages, self.num_mini_batch)]
                data_generators = [rollouts.single_process_recurrent_generator(
                    advantages, self.num_mini_batch, process=i) for i in range(self.num_tasks)]
            elif self.num_tasks > 0:
                assert self.num_tasks == rollouts.num_processes
                data_generators = [rollouts.single_process_feed_forward_generator(
                    advantages, process=i, num_mini_batch=self.num_mini_batch) for i in range(self.num_tasks)]
            else:
                data_generators = [rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)]
            for sample in zip(*data_generators):
                task_losses = []
                for task in range(len(sample)):
                    obs_batch, recurrent_hidden_states_batch, actions_batch, \
                       value_preds_batch, return_batch, masks_batch, attn_masks_batch, old_action_log_probs_batch, \
                            adv_targ = sample[task]

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, dist_l2, _ = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch, attn_masks_batch,
                        actions_batch, attention_act=attention_update)

                    # if attention_update=True, we assume that the log_probs are the attention log_probs.
                    # This is a hack for now, and it is taken care of in model.py and in dual_rl.py
                    ratio = torch.exp(action_log_probs -
                                      old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean()

                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + \
                            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                        value_loss = 0.5 * torch.max(value_losses,
                                                     value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * (return_batch - values).pow(2).mean()
                    task_losses.append(value_loss * self.value_loss_coef + action_loss -
                                       dist_entropy * self.entropy_coef + dist_l2 * self.l2_coef)

                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()
                    dist_entropy_epoch += dist_entropy.item()
                    dist_l2_epoch += dist_l2.item()
                total_loss = torch.stack(task_losses).mean()
                self.optimizer.zero_grad()
                # (value_loss * self.value_loss_coef + action_loss -
                #  dist_entropy * self.entropy_coef).backward()
                # if self.use_pcgrad:
                #     self.optimizer.pc_backward(task_losses)
                # elif self.use_testgrad:
                #     self.optimizer.pc_backward(task_losses)
                # elif self.use_noisygrad:
                #     self.optimizer.noisy_backward(task_losses)
                # elif self.use_median_grad:
                #     self.optimizer.median_backward(task_losses)
                # elif self.use_meanvargrad:
                #     self.optimizer.pc_backward(task_losses)
                # elif self.use_graddrop:
                #     self.optimizer.pc_backward(task_losses)
                # else:
                total_loss.backward()
                # mean_grad, grads, shapes, F_norms_all, F_norms_gru, F_norms_actor, F_norms_critic, F_norms_cat = self.optimizer.plot_backward(task_losses)
                # nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                #                          self.max_grad_norm)
                # if self.attention_policy:
                #     nn.utils.clip_grad_norm_(self.attention_parameters,
                #                              self.max_grad_norm)
                # else:
                #     nn.utils.clip_grad_norm_(self.non_attention_parameters,
                #                              self.max_grad_norm)
                if make_update:
                    self.optimizer.step()
        # last_grad = grads
        # last_mean_grad = mean_grad
        # last_shapes = shapes

                # value_loss_epoch += value_loss.item()
                # action_loss_epoch += action_loss.item()
                # dist_entropy_epoch += dist_entropy.item()
                # dist_l2_epoch += dist_l2.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        dist_l2_epoch /= num_updates

        # return last_mean_grad, last_grad, last_shapes, value_loss_epoch, action_loss_epoch, dist_entropy_epoch, dist_l2_epoch, F_norms_all, F_norms_gru, F_norms_actor, F_norms_critic, F_norms_cat
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, dist_l2_epoch
