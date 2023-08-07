# MultiAgent_PPO with MCTS
An Implementation of PPO for environments with multiple agents with Monte Carlo Tree Search (MCTS). The goal of the agents is to take objects and deliver them to their destinations as quickly as possible. 

# How to run

To train: 

```
python main.py --env-id Warehouse-v0 --grid-size 5 --agents 2 --objects 2 --total-timesteps 50000000 --tensorboard True
```

To show trajectories generated from a model:

```
python main.py --env-id Warehouse-v0 --grid-size 5 --agents 2 --objects 2 --num-envs 8 --num-steps 128 --num-minibatches 1 --update-epochs 4 --total-timesteps 50000000 --tensorboard False --load MODEL_PATH.pt --show
```

# Demo Video
![Screen Recording 2023-07-27 at 17 11 10 (1)](https://github.com/teshnizi/MultiAgent_PPO/assets/48642434/7cb6e1c5-b762-45c7-a404-2cdfe158f827)
