python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --testgrad_beta 0.8 --use_testgrad_median --free_exploration 6 --seed  11 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --testgrad_beta 0.6 --use_testgrad_median --free_exploration 6 --seed  12 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --testgrad_beta 0.4 --use_testgrad_median --free_exploration 6 --seed  13 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --testgrad_beta 0.4 --use_testgrad_median --free_exploration 6 --seed  14 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --testgrad_beta 0.2 --use_testgrad_median --free_exploration 6 --seed  15 &
sleep 3

python3 main.py --env-name "h_bandit-randchoose-v8" --algo  ppo --log-interval 25 --num-steps 100 --num-processes 25 --lr 1e-3 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 3 --num-mini-batch 25 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 6000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --obs_recurrent --use_testgrad --testgrad_beta 0.2 --use_testgrad_median --free_exploration 6 --seed  16 &
wait

echo "obs recurrent 25 arms and free exploration, testgrad_median"
echo "Seed 16 Iter 2000 five_arms 16.0 ten_arms 15.6 many_arms 10.18 , didn't work well"