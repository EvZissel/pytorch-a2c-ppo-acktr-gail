python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_graddrop --seed 1 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_graddrop --seed 6 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_graddrop --seed 2 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_graddrop --seed 3 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_graddrop --seed 4 &

sleep 3

python3 main.py --env-name "h_bandit-randchoose-v5" --algo ppo --log-interval 25 --num-steps 100 --num-processes 10 --lr 3e-4 --entropy-coef 0.05 --value-loss-coef 0.5 --ppo-epoch 1 --num-mini-batch 10 --gamma 0.9 --gae-lambda 0.95 --num-env-steps 5000000 --eval-interval 100 --log-dir ./ppo_log --task_steps=20 --recurrent-policy --use_graddrop --seed 5 &

wait

echo "graddrop recurrent 10 arms."
echo "if graddrop does not hurt, this should show optimal generalization, as without graddrop it is already optimal"