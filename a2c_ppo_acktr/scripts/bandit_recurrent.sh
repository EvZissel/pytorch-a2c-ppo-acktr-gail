nohup python main.py --env-name "h_bandit-obs-randchoose-v8" --algo ppo  \
--log-interval 1 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --l2-coef 0.0 --value-loss-coef 0.5 --weight_decay 0.0 \
--ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 10000000 --eval-interval 10 --eval-nondet_interval 100 \
--log-dir ppo_logs --task_steps=6 --no_normalize --free_exploration 6 --seed 0 --recurrent-policy --save-interval 1000 \
--gpu_device 3 > outputs/bandit_out_25_seed_0_entropy_0.05_recurrent_new.out &
