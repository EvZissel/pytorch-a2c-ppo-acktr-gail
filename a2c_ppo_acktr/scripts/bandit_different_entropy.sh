#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 7000 --l2-coef 0.0 --value-loss-coef 0.5  \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 50000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 0 --obs_recurrent --save-interval 1000 \
#--gpu_device 0 > outputs/bandit_out_25_seed_0_entropy_7000_l2_0.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 8000 --l2-coef 0.0 --value-loss-coef 0.5  \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 50000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 0 --obs_recurrent --save-interval 1000 \
#--gpu_device 0 > outputs/bandit_out_25_seed_0_entropy_8000_l2_0.0.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 9000 --l2-coef 0.0 --value-loss-coef 0.5  \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 50000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 0 --obs_recurrent --save-interval 1000 \
#--gpu_device 0 > outputs/bandit_out_25_seed_0_entropy_9000_l2_0.0.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 11000 --l2-coef 0.0 --value-loss-coef 0.5  \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 50000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 0 --obs_recurrent --save-interval 1000 \
#--gpu_device 0 > outputs/bandit_out_25_seed_0_entropy_11000_l2_0.0.out &
#
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 12000 --l2-coef 0.0 --value-loss-coef 0.5  \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 50000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 0 --obs_recurrent --save-interval 1000 \
#--gpu_device 0 > outputs/bandit_out_25_seed_0_entropy_12000_l2_0.0.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 13000 --l2-coef 0.0 --value-loss-coef 0.5  \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 50000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 0 --obs_recurrent --save-interval 1000 \
#--gpu_device 0 > outputs/bandit_out_25_seed_0_entropy_13000_l2_0.0.out &

#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --l2-coef 10000 --value-loss-coef 0.5  \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 0 --obs_recurrent --save-interval 1000 \
#--gpu_device 0 --continue_from_epoch 9999 --save-dir ./ppo_logs/runs/h_bandit-obs-randchoose-v8_ppo_seed_0_num_arms_25_entro_0.05_l2_10000.0_29-08-2021_01-30-44 \
#> outputs/bandit_out_25_seed_0_entropy_0.05_l2_10000_cont.out &

#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.0 --l2-coef 0.0001 --value-loss-coef 0.5 --weight_decay 0.0 \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
#--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.0_l2_0.0001.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.0 --l2-coef 0.001 --value-loss-coef 0.5 --weight_decay 0.0 \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
#--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.0_l2_0.001.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.0  --l2-coef 0.01 --value-loss-coef 0.5 --weight_decay 0.0 \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
#--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.0_l2_0.01.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.0 --l2-coef 0.05 --value-loss-coef 0.5 --weight_decay 0.0 \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
#--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.0_l2_0.05.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.0 --l2-coef 0.1 --value-loss-coef 0.5 --weight_decay 0.0 \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
#--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.0_l2_0.1.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.0 --l2-coef 0.2 --value-loss-coef 0.5 --weight_decay 0.0 \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
#--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.0_l2_0.2.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.0 --l2-coef 0.3 --value-loss-coef 0.5 --weight_decay 0.0 \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
#--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.0_l2_0.3.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.0 --l2-coef 0.4 --value-loss-coef 0.5 --weight_decay 0.0 \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 7962 --obs_recurrent --save-interval 1000 \
#--gpu_device 1 > outputs/bandit_out_25_seed_7962_entropy_0.0_l2_0.4.out &

sleep 3

nohup python main.py --env-name "h_bandit-obs-randchoose-v14" --algo ppo  \
--log-interval 1 --num-steps 6 --num-processes 400 --lr 1e-3 --entropy-coef 0.05 --l2-coef 0.0 --value-loss-coef 0.5 --weight_decay 0.0 \
--ppo-epoch 3 --num-mini-batch 400 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 --eval-nondet_interval 100 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 0 --obs_recurrent --save-interval 1000 \
--gpu_device 2 > outputs/bandit_out_400_seed_0.out &

sleep 3

nohup python main.py --env-name "h_bandit-obs-randchoose-v14" --algo ppo  \
--log-interval 1 --num-steps 6 --num-processes 400 --lr 1e-3 --entropy-coef 0.05 --l2-coef 0.0 --value-loss-coef 0.5 --weight_decay 0.0 \
--ppo-epoch 3 --num-mini-batch 400 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 --eval-nondet_interval 100 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 456 --obs_recurrent --save-interval 1000 \
--gpu_device 2 > outputs/bandit_out_400_seed_456.out &

sleep 3

nohup python main.py --env-name "h_bandit-obs-randchoose-v14" --algo ppo  \
--log-interval 1 --num-steps 6 --num-processes 400 --lr 1e-3 --entropy-coef 0.05 --l2-coef 0.0 --value-loss-coef 0.5 --weight_decay 0.0 \
--ppo-epoch 3 --num-mini-batch 400 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 --eval-nondet_interval 100 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 9687435 --obs_recurrent --save-interval 1000 \
--gpu_device 2 > outputs/bandit_out_400_seed_9687435.out &

sleep 3

nohup python main.py --env-name "h_bandit-obs-randchoose-v14" --algo ppo  \
--log-interval 1 --num-steps 6 --num-processes 400 --lr 1e-3 --entropy-coef 0.05 --l2-coef 0.0 --value-loss-coef 0.5 --weight_decay 0.0 \
--ppo-epoch 3 --num-mini-batch 400 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 --eval-nondet_interval 100 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 65412 --obs_recurrent --save-interval 1000 \
--gpu_device 2 > outputs/bandit_out_400_seed_65412.out &
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.4 --value-loss-coef 0.5  \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 0 --recurrent --save-interval 1000 \
#--gpu_device 3 > outputs/bandit_out_25_seed_0_entropy_0.4_recurrent.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.5 --value-loss-coef 0.5  \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 0 --recurrent --save-interval 1000 \
#--gpu_device 3 > outputs/bandit_out_25_seed_0_entropy_0.5_recurrent.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.6 --value-loss-coef 0.5  \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 0 --recurrent --save-interval 1000 \
#--gpu_device 3 > outputs/bandit_out_25_seed_0_entropy_0.6_recurrent.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.8 --value-loss-coef 0.5  \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 25000000 --eval-interval 10 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 0 --recurrent --save-interval 1000 \
#--gpu_device 3 > outputs/bandit_out_25_seed_0_entropy_0.8_recurrent.out &
#
#sleep 3
#
#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 1.0 --value-loss-coef 0.5  \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 5 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 354 --obs_recurrent --save-interval 1000 \
#--gpu_device 3 > outputs/bandit_out_25_seed_1235_entropy_1.out &