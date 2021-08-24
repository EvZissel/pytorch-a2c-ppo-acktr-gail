nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.01 --value-loss-coef 0.5  \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.01.out &

sleep 3

nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.001 --value-loss-coef 0.5  \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.001.out &


sleep 3

nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.0001 --value-loss-coef 0.5  \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.0001.out &

sleep 3

nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5  \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.05.out &


sleep 3

nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.2 --value-loss-coef 0.5  \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.2.out &

sleep 3

nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.3 --value-loss-coef 0.5  \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.3.out &

sleep 3

nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.4 --value-loss-coef 0.5  \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.4.out &

sleep 3

nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.5 --value-loss-coef 0.5  \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.5.out &

sleep 3

nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.6 --value-loss-coef 0.5  \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.6.out &

sleep 3

nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.8 --value-loss-coef 0.5  \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.8.out &

#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 1.0 --value-loss-coef 0.5  \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 354 --obs_recurrent --save-interval 1000 \
#--gpu_device 3 > outputs/bandit_out_25_seed_1235_entropy_1.out &