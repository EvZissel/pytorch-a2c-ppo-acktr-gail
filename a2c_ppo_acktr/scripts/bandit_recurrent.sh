#nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
#--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --l2-coef 0.0 --value-loss-coef 0.5 --weight_decay 0.0 \
#--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 10 --eval-nondet_interval 100 \
#--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 0 --recurrent-policy --save-interval 1000 \
#--gpu_device 3 > outputs/bandit_out_25_seed_0_entropy_0.05_recurrent_new.out &

# same as before
CUDA_VISIBLE_DEVICES=1 nohup python main_dqn.py --env "h_bandit-randchoose-v8" --num-processes 25 --num-steps 100 --seed 0 --task_steps 6 \
 --free_exploration 6 --recurrent-policy --no_normalize --log-dir dqn_logs --num-mini-batch 25 --learning_rate 0.0001 --epsilon_start 1.0 \
--epsilon_end 0.01 --epsilon_decay 3000000 --max_ts 40000000 --gamma 0.99 --log_every 1 --target_network_update_f 4 > outputs/dqn_bandit_out_25_seed_0_recurrent.out &

sleep 3

# large epsilon end
CUDA_VISIBLE_DEVICES=1 nohup python main_dqn.py --env "h_bandit-randchoose-v8" --num-processes 25 --num-steps 100 --seed 0 --task_steps 6 \
 --free_exploration 6 --recurrent-policy --no_normalize --log-dir dqn_logs --num-mini-batch 25 --learning_rate 0.0001 --epsilon_start 1.0 \
--epsilon_end 0.1 --epsilon_decay 3000000 --max_ts 40000000 --gamma 0.99 --log_every 1 --target_network_update_f 4 > outputs/dqn_bandit_out_25_seed_0_recurrent_epsilon_end0.1.out &

sleep 3

# small epsilon decay
CUDA_VISIBLE_DEVICES=1 nohup python main_dqn.py --env "h_bandit-randchoose-v8" --num-processes 25 --num-steps 100 --seed 0 --task_steps 6 \
 --free_exploration 6 --recurrent-policy --no_normalize --log-dir dqn_logs --num-mini-batch 25 --learning_rate 0.0001 --epsilon_start 1.0 \
--epsilon_end 0.01 --epsilon_decay 30000000 --max_ts 40000000 --gamma 0.99 --log_every 1 --target_network_update_f 4 > outputs/dqn_bandit_out_25_seed_0_recurrent_epsilon_decay30000000.out &

sleep 3

# larger lr
CUDA_VISIBLE_DEVICES=1 nohup python main_dqn.py --env "h_bandit-randchoose-v8" --num-processes 25 --num-steps 100 --seed 0 --task_steps 6 \
 --free_exploration 6 --recurrent-policy --no_normalize --log-dir dqn_logs --num-mini-batch 25 --learning_rate 0.001 --epsilon_start 1.0 \
--epsilon_end 0.01 --epsilon_decay 3000000 --max_ts 40000000 --gamma 0.99 --log_every 1 --target_network_update_f 4 > outputs/dqn_bandit_out_25_seed_0_recurrent_lr0.001.out &

sleep 3

# large lr
CUDA_VISIBLE_DEVICES=1 nohup python main_dqn.py --env "h_bandit-randchoose-v8" --num-processes 25 --num-steps 100 --seed 0 --task_steps 6 \
 --free_exploration 6 --recurrent-policy --no_normalize --log-dir dqn_logs --num-mini-batch 25 --learning_rate 0.01 --epsilon_start 1.0 \
--epsilon_end 0.01 --epsilon_decay 3000000 --max_ts 40000000 --gamma 0.99 --log_every 1 --target_network_update_f 4 > outputs/dqn_bandit_out_25_seed_0_recurrent_lr0.01.out &

sleep 3

# very larger lr
CUDA_VISIBLE_DEVICES=1 nohup python main_dqn.py --env "h_bandit-randchoose-v8" --num-processes 25 --num-steps 100 --seed 0 --task_steps 6 \
 --free_exploration 6 --recurrent-policy --no_normalize --log-dir dqn_logs --num-mini-batch 25 --learning_rate 0.1 --epsilon_start 1.0 \
--epsilon_end 0.01 --epsilon_decay 3000000 --max_ts 40000000 --gamma 0.99 --log_every 1 --target_network_update_f 4 > outputs/dqn_bandit_out_25_seed_0_recurrentlr0.1.out &