from __future__ import annotations


# -----Learning Parameters---
alpha = 0.1    # learning rate
gamma = 0.7    # importance of next action 0.9
epsilon = 1.0 # exploration chance was 0.1
epsilon_min = 0.095
epsilon_decay = 0.999993

# ------Reward and Punishment----
STEAL_POTION = 9
CAUGHT_BY_PLAYER = -10
MOVE_REWARD = -0.1

# ---------Map configurations------------
w = 80
h = 50
# determine how many directions can agent moves.
directions = 4   # you may change it to 4: up,down,left and right.
radius = 3

target = "Health Potion"
# training
#predator = "Miner"
#prey = "Player"

predator = "Player"
prey = "Miner"
