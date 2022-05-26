import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--val_lr', type=float, default=7e-4, help='validation agent learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--start-level', type=int, default=0, help='start level (default: 0)')
    parser.add_argument(
        '--num-level', type=int, default=128, help='num level (default: 25)')
    parser.add_argument(
        '--distribution-mode', type=str, default='easy', help='distribution mode for environment')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=128,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=512,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=1,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10e3,
        help='save interval, one save per n updates (default: 10e2)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--eval-nondet_interval',
        type=int,
        default=None,
        help='eval interval non-deterministic, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='maze',
        help='environment to train on (default: maze)')
    parser.add_argument(
        '--val_env_name',
        default='maze',
        help='environment to train on (default: maze)')
    parser.add_argument(
        '--val_agent_steps',
        type=int,
        default=1,
        help='number of PPO steps to train val agent for each agent update')
    parser.add_argument(
        '--val_reinforce_update',
        action='store_true',
        default=False,
        help='compute validation attention using reinforce')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default="",
        help='directory to save agent logs (default: "")')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--gpu_device',
        type=int,
        default = int(0),
        required=False,
        help = 'visible device in CUDA')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--obs_recurrent',
        action='store_true',
        default=False,
        help='use a recurrent policy and observations input')
    parser.add_argument(
        '--recurrent-hidden-size',
        type=int,
        default=int(256),
        required=False,
        help='GRU hidden layer size')
    parser.add_argument(
        '--reinitialization',
        action='store_true',
        default=False,
        help='reinitialize GRU and the last few layers')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    parser.add_argument(
        '--use_pcgrad',
        action='store_true',
        default=False,
        help='use pcgrad in ppo')
    parser.add_argument(
        '--use_testgrad',
        action='store_true',
        default=False,
        help='use testgrad in ppo')
    parser.add_argument(
        '--use_testgrad_median',
        action='store_true',
        default=False,
        help='use testgrad with median gradient instead of mean in ppo')
    parser.add_argument(
        '--use_median_grad',
        action='store_true',
        default=False,
        help='use median gradient + noise instead of mean in ppo')
    parser.add_argument(
        '--use_noisygrad',
        action='store_true',
        default=False,
        help='use noisygrad in ppo')
    parser.add_argument(
        '--use_graddrop',
        action='store_true',
        default=False,
        help='use graddrop in ppo')
    parser.add_argument(
        '--use_privacy',
        action='store_true',
        default=False,
        help='use differentially private SGD in ppo')
    parser.add_argument(
        '--continue_from_epoch',
        type=int,
        default=0,
        help='load previous training (from model save dir) and continue')
    parser.add_argument(
        '--max_task_grad_norm',
        type=float,
        default=1000.0,
        help='per-task or per-sample gradient clipping in noisy_grad and dp_sgd')
    parser.add_argument(
        '--grad_noise_ratio',
        type=float,
        default=0.0,
        help='gradient noise ratio for noisy_grad and dp_sgd')
    parser.add_argument(
        '--testgrad_quantile',
        type=float,
        default=-1.0,
        help='quantile gradient in testgrad (float on [0,1])')
    parser.add_argument(
        '--task_steps',
        type=int,
        default=20,
        help='number of steps in each task')
    parser.add_argument(
        '--free_exploration',
        type=int,
        default=0,
        help='number of steps in each task without reward')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='weight decay in Adam')
    parser.add_argument(
        '--testgrad_alpha',
        type=float,
        default=1.0,
        help='alpha mixing parameter in testgrad')
    parser.add_argument(
        '--testgrad_beta',
        type=float,
        default=1.0,
        help='beta threshold parameter in testgrad (fraction of sign agreements)')
    parser.add_argument(
        '--no_special_grad_for_critic',
        action='store_true',
        default=False,
        help='no special grad for critic in testgrad')
    parser.add_argument(
        '--use_meanvargrad',
        action='store_true',
        default=False,
        help='use mean-variance gradient instead of mean in ppo')
    parser.add_argument(
        '--meanvar_beta',
        type=float,
        default=1.0,
        help='beta threshold parameter in mean - beta*var')
    parser.add_argument(
        '--val_improvement_threshold',
        type=float,
        default=-1e6,
        help='threshold on improvement in validation reward for reverting the gradient update')
    parser.add_argument(
        '--hard_attn',
        action='store_true',
        default=False,
        help='use hard attention in val agent')
    parser.add_argument(
        '--att_size',
        type=int,
        default=8,
        help='attention size')
    parser.add_argument(
        '--no_normalize',
        action='store_true',
        default=False,
        help='no normalize inputs')
    parser.add_argument(
        '--attention_features',
        action='store_true',
        default=False,
        help='attention on the feature space')
    parser.add_argument(
        '--deterministic_attention',
        action='store_true',
        default=False,
        help='deterministic attention for policy update')
    parser.add_argument(
        '--det_eval_attention',
        action='store_true',
        default=False,
        help='deterministic attention for evaluation')
    parser.add_argument(
        '--normalize_rew',
        action='store_true',
        default=False,
        help='normalize reword')
    parser.add_argument(
        '--mask_size',
        type=int,
        default=0,
        help='constant mask size')
    parser.add_argument(
        '--mask_all',
        action='store_true',
        default=False,
        help='mask all frame')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
