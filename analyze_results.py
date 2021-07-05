import os
import sys

import torch

import matplotlib.pyplot as plt
import numpy as np

from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.utils import save_obj, load_obj

EVAL_ENVS = {'three_arms': 'h_bandit-randchoose-v0',
             'five_arms': 'h_bandit-randchoose-v5',
             'many_arms': 'h_bandit-randchoose-v1'}

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # logdir = os.path.join(os.path.expanduser(args.log_dir), 'runs')
    logdir = os.path.expanduser(args.log_dir)

    # Ugly but simple logging
    log_dict = {
        'task_steps': args.task_steps,
        'grad_noise_ratio': args.grad_noise_ratio,
        'max_task_grad_norm': args.max_task_grad_norm,
        'use_noisygrad': args.use_noisygrad,
        'use_pcgrad': args.use_noisygrad,
        'use_privacy': args.use_privacy,
        'seed': args.seed,
        'cmd': ' '.join(sys.argv[1:])
    }
    for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        log_dict[eval_disp_name] = []


        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     actor_critic.eval()
        #     obs_rms = utils.get_vec_normalize(envs).obs_rms
        #     eval_r = {}
        #     for eval_disp_name, eval_env_name in EVAL_ENVS.items():
        #         print(eval_disp_name)
        #         eval_r[eval_disp_name] = evaluate(actor_critic, obs_rms, eval_env_name, args.seed,
        #                                           args.num_processes, logdir, device, steps=args.task_steps)
        #         summary_writer.add_scalar(f'eval/{eval_disp_name}', eval_r[eval_disp_name], (j+1) * args.num_processes * args.num_steps)
        #         log_dict[eval_disp_name].append([(j+1) * args.num_processes * args.num_steps, eval_r[eval_disp_name]])
        #     summary_writer.add_scalars('eval_combined', eval_r, (j+1) * args.num_processes * args.num_steps)
        #     actor_critic.train()

    res_search = [
        # [{'use_testgrad': True,
        #   'task_steps': 20,
        #   'grad_noise_ratio': 0.03}, 'testgrad 0.03'],
        # [{'use_testgrad': True,
        #   'task_steps': 20,
        #   'grad_noise_ratio': 1.0}, 'testgrad 1.0 '],
        [{'task_steps': 6}, 'baseline'],
    ]

    for s in res_search:
        res_train = []
        res_test = []
        res_type = []
        for subdir, dirs, files in os.walk(logdir):
            for name in dirs:
                load_name = os.path.join(logdir, name, 'log_dict.pkl')
                try:
                    log_dict = load_obj(load_name)
                except:
                    continue
                is_match = True
                for key, val in s[0].items():
                    if log_dict[key] != val:
                        is_match = False
                if is_match:
                    res_train.append(log_dict['partial_train_eval'])
                    res_test.append(log_dict['test_eval'])
                    res_type.append(log_dict['cmd'])

        if len(res_test) > 0:
            for i in range(len(res_test)):
                for j in range(len(res_test[i])):
                    res_test[i][j][1] = np.mean(np.array(res_test[i][j][1]))
                    res_train[i][j][1] = np.mean(np.array(res_train[i][j][1]))
            res_test = np.array(res_test)
            res_train = np.array(res_train)
            t = res_test[0, :, 0]
            res_test_mean = np.mean(res_test[:, :, 1], axis=0)
            res_test_std = np.std(res_test[:, :, 1], axis=0)
            res_train_mean = np.mean(res_train[:, :, 1], axis=0)
            res_train_std = np.std(res_train[:, :, 1], axis=0)
            plt.errorbar(t, res_test_mean, res_test_std, label=s[1])
            plt.errorbar(t, res_train_mean, res_train_std, label=s[1])
            # for i in range(res_test.shape[0]):
            #     plt.plot(res_test[i,:,0], res_test[i,:,1], label=s[1])
            #     plt.plot(res_train[i, :, 0], res_train[i, :, 1], label=s[1])
    plt.legend()
    plt.show()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
