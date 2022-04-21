# Modified code from
# DQN implementation by Tycho van der Ouderaa found at
# https://github.com/tychovdo/PacmanDQN

# Used code from
# DQN implementation by Tejas Kulkarni found at
# https://github.com/mrkulk/deepQN_tensorflow

# Used code from:
# The Pacman AI projects were developed at UC Berkeley found at
# http://ai.berkeley.edu/project_overview.html

# Modified code from
# Q-learning implementation by Mazzy Star found at
# https://github.com/mazzzystar/QLearningMouse


from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple
import random


import numpy as np
import tcod

from actions import Action, BumpAction, MeleeAction, MovementAction, WaitAction
# for bfs agent
from queue import Queue
from actions import PickupAction
from entity import Item
import components.qlearn as qlearn
import components.config as cfg
from tcod.map import compute_fov

# For dqn agent
import time
import sys

# Replay memory
from collections import deque

# Neural nets
import tensorflow as tf
import components.dqn as dqn

if TYPE_CHECKING:
    from entity import Actor


class BaseAI(Action):
    is_dqn = False
    def perform(self) -> None:
        raise NotImplementedError()

    def get_path_to(self, dest_x: int, dest_y: int) -> List[Tuple[int, int]]:
        """Compute and return a path to the target position.

        If there is no valid path then returns an empty list.
        """
        # Copy the walkable array.
        cost = np.array(self.entity.gamemap.tiles["walkable"], dtype=np.int8)

        for entity in self.entity.gamemap.entities:
            # Check that an enitiy blocks movement and the cost isn't zero (blocking.)
            if entity.blocks_movement and cost[entity.x, entity.y]:
                # Add to the cost of a blocked position.
                # A lower number means more enemies will crowd behind each other in
                # hallways.  A higher number means enemies will take longer paths in
                # order to surround the player.
                cost[entity.x, entity.y] += 10

        # Create a graph from the cost array and pass that graph to a new pathfinder.
        graph = tcod.path.SimpleGraph(cost=cost, cardinal=2, diagonal=3)
        pathfinder = tcod.path.Pathfinder(graph)

        pathfinder.add_root((self.entity.x, self.entity.y))  # Start position.

        # Compute the path to the destination and remove the starting point.
        path: List[List[int]] = pathfinder.path_to((dest_x, dest_y))[1:].tolist()

        # Convert from List[List[int]] to List[Tuple[int, int]].
        return [(index[0], index[1]) for index in path]

    def get_moves(self):
        return [
            #(-1, -1),  # Northwest
            (-1, 0),  # West
            #(-1, 1),  # Southwest

            (0, -1),  # North
            (0, 1),  # South

            #(1, -1),  # Northeast
            (1, 0),  # East
            #(1, 1),  # Southeast
        ]


    def get_value(self, mdict, key):
        try:
            return mdict[key]
        except KeyError:
            return 0

# for q n dqn agents
    def reset_pos(self) -> None:
        # reset to a safe place

        predator_coords = []
        for ent in self.entity.gamemap.entities:
            if ent.name == cfg.predator:
                # get all predator coords
                predator_coords.append([ent.x,ent.y])

        grid = np.array(self.entity.gamemap.tiles["walkable"], dtype=np.int8)

        while True:
            reset_x, reset_y = random.choice(np.argwhere(np.array(grid)==1))
            # ensure it is not 0,0
            if not(reset_x == 0 and reset_y == 0):
                for pos in predator_coords:
                    # ensure new pos is safe
                    if abs(pos[0]-reset_x) <=1 and abs(pos[1]-reset_y)<=1:

                        break

                break

        self.entity.x = reset_x
        self.entity.y = reset_y

 # for competitive enemy with 4 directions
    def get_path_bfs(self, dest_x: int, dest_y: int) -> List[Tuple[int, int]]:
        grid_list = np.array(self.entity.gamemap.tiles["walkable"], dtype=np.int8)
        #print(grid_list)

        fh = len(grid_list)
        fw = max([len(x) for x in grid_list])

        start = (self.entity.x, self.entity.y)
        end = (dest_x,dest_y)

        moves = self.get_moves()
        for n in moves:
            if (start[0]+n[0],start[1]+n[1]) == end:
               # if next move can go towards target
                return [n]

        best_action = None
        q = Queue()

        q.put(start)
        step = 1
        V = {}
        preV = {}

        V[start] = 0

        while not q.empty():

            grid = q.get()

            for i in range(len(moves)):

                ny, nx = grid[0] + moves[i][0], grid[1] + moves[i][1]

                if nx < 0 or ny < 0 or nx > (fw-1) or ny > (fh-1):
                    continue

                if self.get_value(V, (ny, nx)) or grid_list[ny][nx] == 0:  # has visit or is wall.

                    continue

                preV[(ny, nx)] = self.get_value(V, (grid[0], grid[1]))

                if ny == end[0] and nx == end[1]:
                    V[(ny, nx)] = step + 1
                    seq = []
                    last = V[(ny, nx)]
                    while last > 1:
                        k = [key for key in V if V[key] == last]
                        seq.append(k[0])
                        assert len(k) == 1
                        last = preV[(k[0][0], k[0][1])]
                    seq.reverse()


                    best_action = (seq[0][0]-start[0],seq[0][1]-start[1])

                q.put((ny, nx))
                step += 1
                V[(ny, nx)] = step

        if best_action is not None:

            return [best_action]

        else:
            # no best action found, return a random action

            return [random.choice(self.get_moves())]


# ===================================================================================================
class HostileEnemy(BaseAI):
    def __init__(self, entity: Actor):
        super().__init__(entity)
        self.path: List[Tuple[int, int]] = []
        self.won = False
        self.lost = False

    def perform(self) -> None:
        target = self.engine.player
        dx = target.x - self.entity.x
        dy = target.y - self.entity.y
        # attack when 1 or 0
        distance = max(abs(dx), abs(dy))  # Chebyshev distance.

        if self.engine.game_map.visible[self.entity.x, self.entity.y]:
            if distance <= 1:
                return MeleeAction(self.entity, dx, dy).perform()

            self.path = self.get_path_to(target.x, target.y)

        if self.path:
            dest_x, dest_y = self.path.pop(0)
            # 4 actions
            return MovementAction(
                self.entity,
                dest_x - self.entity.x,
                dest_y - self.entity.y,
            ).perform()

        return WaitAction(self.entity).perform()


class ConfusedEnemy(BaseAI):
    """
    A confused enemy will stumble around aimlessly for a given number of turns, then revert back to its previous AI.
    If an actor occupies a tile it is randomly moving into, it will attack.
    """

    def __init__(self, entity: Actor, previous_ai: Optional[BaseAI], turns_remaining: int):
        super().__init__(entity)

        self.previous_ai = previous_ai
        self.turns_remaining = turns_remaining

    def perform(self) -> None:
        # Revert the AI back to the original state if the effect has run its course.
        if self.turns_remaining <= 0:
            self.engine.message_log.add_message(f"The {self.entity.name} is no longer confused.")
            self.entity.ai = self.previous_ai
        else:
            # Pick a random direction
            direction_x, direction_y = random.choice(self.get_moves())

            self.turns_remaining -= 1

            # The actor will either try to move or attack in the chosen random direction.
            # Its possible the actor will just bump into the wall, wasting a turn.
            return BumpAction(
                self.entity,
                direction_x,
                direction_y,
            ).perform()


# Added agent for training: chases thief2 agent
class CompetitiveEnemy(BaseAI):
        def __init__(self, entity: Actor):
            super().__init__(entity)
            self.path: List[Tuple[int, int]] = []

        def perform(self) -> None:
            target = self.engine.player

            if self.engine.game_map.visible[self.entity.x, self.entity.y]:

                self.path = self.get_path_bfs(target.x,target.y)

            if self.path:
                x,y =self.path[0][0],self.path[0][1]

                return MovementAction(
                    self.entity,
                    x,y,
                ).perform()

            direction_x, direction_y = random.choice(self.get_moves())

            return MovementAction(
                self.entity,
                direction_x, direction_y,
            ).perform()

# Random agent with targets
# stealing the items in the dungeon, but does not attack the player
class ThiefEnemy(BaseAI):
    def __init__(self, entity: Actor):
        super().__init__(entity)
        self.path: List[Tuple[int, int]] = []
        self.thiefScore = 0
        self.playerScore = 0
        self.reward = 0
        self.new_round = False
        self.player_lose = False
        self.player_win = False
        print('-------random agent------')

    def won(self):
        return self.player_win

    def lost(self):
        return self.player_lose

    def is_new_round(self) -> None:
        return self.new_round

    def get_score(self) -> None:
        return self.thiefScore
    def get_rounds(self) -> None:
        return self.thiefScore+self.playerScore

    def perform(self) -> None:
        pickup = False
        beaten = False
        self.new_round = False
        self.reward += cfg.MOVE_REWARD


        for target in self.entity.gamemap.entities:
            if target.name == cfg.predator:
                if (abs(self.entity.x-target.x)==1 and abs(self.entity.y-target.y)==0) or (abs(self.entity.x-target.x)==0 and abs(self.entity.y-target.y)==1):

                    self.playerScore += 1
                    self.reward += cfg.CAUGHT_BY_PLAYER
                    beaten = True
                    break

            if self.engine.game_map.visible[target.x, target.y]:
            # if there is an item, find the path to the item
                if target.name == cfg.target:
                    # check if the current position is an item
                    if (self.entity.x, self.entity.y) == (target.x, target.y):
                        #print("Stealer picked up", target.name)
                        self.thiefScore += 1
                        pickup = True
                        self.reward += cfg.STEAL_POTION

                        break

                    self.path = self.get_path_to(target.x, target.y)
        end = True
        print('AI hp: ',self.entity.fighter.hp)
        print('Your hp: ',self.engine.player.fighter.hp)
        # check if all hps are collected
        for i in self.entity.gamemap.entities:
            if i.name == cfg.target:
                end = False
        if end:
            c = 0
            for i in self.engine.player.inventory.items:
                if i.name == cfg.target:
                    c+=1
            if self.thiefScore > c:
                self.player_lose = True

            elif self.thiefScore < c:
                self.player_win = True

            else:
                if self.engine.player.fighter.hp > self.entity.fighter.hp:
                    self.player_win = True

        if pickup or beaten:
            self.new_round = True
        #    f1 = open('astar_reward.txt', 'a')
        #    f1.write(str(self.reward)+'\n')

        #    f1.close()
            self.reward = 0

            if pickup:

                return PickupAction(self.entity).perform()


        if self.path:
            dest_x, dest_y = self.path.pop(0)
            direction_x, direction_y = dest_x - self.entity.x, dest_y - self.entity.y

        #move to the target or random moves if there is no target
        else:
            direction_x, direction_y = random.choice(self.get_moves())

        return MovementAction(
            self.entity,
            direction_x, direction_y,
        ).perform()


# Added agent:
class ThiefEnemy2(BaseAI):
    def __init__(self, entity: Actor):
        super().__init__(entity)
        #self.path: List[Tuple[int, int]] = []
        self.ai = qlearn.QLearn(actions=range(cfg.directions), alpha=0.1, gamma=cfg.gamma, epsilon=0)
        self.lastState = None
        self.lastAction = None
        self.new_round = False
        self.thiefScore = 0
        self.playerScore = 0
        self.player_lose = False
        self.player_win = False
        print('-------qlearn agent------')

    def won(self):
        return self.player_win

    def lost(self):
        return self.player_lose

    def calculate_state(self):
        def cell_value(x, y):

            for ent in self.entity.gamemap.entities:
                # TODO: change miner to player later
                if ent.name == cfg.predator and (x == ent.x and y == ent.y):
                    return 3
                if ent.name == cfg.target and (x == ent.x and y == ent.y):
                    return 2
            if not self.engine.game_map.in_bounds(x, y):
                # Destination is out of bounds.
                return 1
            if not self.engine.game_map.tiles["walkable"][x, y]:
                # Destination is blocked by a tile.
                return 1
            if self.engine.game_map.get_blocking_entity_at_location(x, y):
                # Destination is blocked by an entity.
                return 1
            return 0

        # for x,y in all visible states:
        # get player's position
        for ent in self.entity.gamemap.entities:
            if ent.name == cfg.prey:
                center_x, center_y = ent.x, ent.y

        n = 2*cfg.radius+1
        grid = np.ones((n,n,2))
        # range from center_x-r to center_x+r
        i = 0

        for x in range(center_x-cfg.radius, center_x+cfg.radius+1):
            j = 0
            for y in range(center_y-cfg.radius, center_y+cfg.radius+1):
                grid[i][j] = (x,y)
                j+=1
            i+=1

        # remove center cell
        grid = np.delete(grid.reshape([n*n,2]), n*n//2, axis=0)
        return tuple([cell_value(int(dir[0]), int(dir[1])) for dir in grid])

    def save_agent(self) -> None:
        self.ai.save_qtable()

    # for training
    def get_score(self) -> None:
        return self.thiefScore

    def get_rounds(self) -> None:
        return self.thiefScore+self.playerScore

    def is_new_round(self) -> None:
        return self.new_round


    def perform(self) -> None:

        state = self.calculate_state()
        pickup = False
        beaten = False
        self.new_round = False

        reward = cfg.MOVE_REWARD

        for ent in self.entity.gamemap.entities:
            if ent.name == cfg.predator:
                if (abs(self.entity.x-ent.x)==0 and abs(self.entity.y-ent.y)==1) or (abs(self.entity.x-ent.x)==1 and abs(self.entity.y-ent.y)==0):
                    reward = cfg.CAUGHT_BY_PLAYER
                    if self.lastState is not None:
                        self.ai.learn(self.lastState, self.lastAction, state, reward)
                    self.lastState = None
                    beaten = True
                    self.playerScore += 1
                    break
            if ent.name == cfg.target and (self.entity.x == ent.x and self.entity.y == ent.y):

                reward = cfg.STEAL_POTION
                pickup = True
                self.thiefScore += 1
                break

        if not beaten:
            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, state, reward)

        end = True
        print('AI hp: ',self.entity.fighter.hp)
        print('Your hp: ',self.engine.player.fighter.hp)
        # check if all hps are collected
        for i in self.entity.gamemap.entities:
            if i.name == cfg.target:
                end = False
        if end:
            c = 0
            for i in self.engine.player.inventory.items:
                if i.name == cfg.target:
                    c+=1
            if self.thiefScore > c:
                self.player_lose = True

            elif self.thiefScore < c:
                self.player_win = True

            else:
                if self.engine.player.fighter.hp > self.entity.fighter.hp:
                    self.player_win = True

        if pickup or beaten:
            # round ended
            self.new_round = True
            #f = open('gamma/agent_reward_d4_g0.7.txt', 'a')
            #f.write(str(reward)+'\n')
            #f.close()
            reward = 0
            if pickup:
                return PickupAction(self.entity).perform()

            # choose a new action and execute it
        action = self.ai.choose_action(state) # move to main
        self.lastState = state
        self.lastAction = action # a number
            # 0: left, 1: left down, 2: down, 3: right down, 4: right, 5: right up, 6: up, 7: left up
        dir_x, dir_y = self.get_moves()[action]

        return MovementAction(
                self.entity,
                dir_x,
                dir_y,
            ).perform()

# dqn agent
class ThiefEnemy3(BaseAI):
    is_dqn = True
    def __init__(self, entity: Actor):
        super().__init__(entity)

        self.params = {
            # Model backups
            'load_file': tf.compat.v1.train.latest_checkpoint('saves/'),

            # Training parameters
            'train_start': 10000,    # Episodes before training starts #was 10000
            'batch_size': 32,       # Replay memory batch size # minimum
            'mem_size': 100000,     # Replay memory size #maximum size

            'discount': 0.95,       # Discount rate (gamma value)
            'lr': .0002,            # Learning reate
            # 'rms_decay': 0.99,      # RMS Prop decay (switched to adam)
            # 'rms_eps': 1e-6,        # RMS Prop epsilon (switched to adam)

            # Epsilon value (epsilon-greedy)
            'eps': 0,             # Epsilon start value 1.0
            'eps_final': 0,       # Epsilon end value 0.1
            'eps_step': 1600000,    # Epsilon steps between start and end (linear) #was 34000

            # fov width and height of map
            'width': 2*cfg.radius+1,
            'height': 2*cfg.radius+1,

            'dueling': True,
            'history': True,
            'mlp': True,

        }

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)

        # error from session
        self.sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))
        self.qnet = dqn.DQN(self.params,'qnet')

        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        # Q and cost
        self.Q_global = []
        self.cost_disp = 0

        # Stats
        self.cnt = self.qnet.sess.run(self.qnet.global_step)

        self.local_cnt = 0

        self.numeps = 0

        self.s = time.time()

        self.replay_mem = deque()

        self.rounds = 0
        self.score = 0
        self.new_round = True
        self.player_lose = False
        self.player_win = False

    def won(self):
        #print('win')
        return self.player_win

    def lost(self):
        #print('win')
        return self.player_lose

    def get_rounds(self) -> None:
        return self.rounds

    def is_new_round(self) -> None:
        return self.new_round

    def get_score(self) -> None:
        return self.score

    def getMove(self):
        # Exploit / Explore

        if np.random.rand() > self.params['eps']:
            #print('Exploit action')
            if self.params['history']:
                states = self.still_4frames
                n = 16
            else:
                states = self.current_state
                n = 4
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y,
                feed_dict = {self.qnet.x: np.reshape(states,
                                                     (1, self.params['width'], self.params['height'], n)),
                             self.qnet.q_t: np.zeros(1),
                             self.qnet.actions: np.zeros((1, 4)),
                             self.qnet.terminals: np.zeros(1),
                             self.qnet.rewards: np.zeros(1)})[0]


            self.Q_global.append(max(self.Q_pred))
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

            if len(a_winner) > 1:
                move = self.get_moves()[a_winner[np.random.randint(0, len(a_winner))][0]]
            else:
                move = self.get_moves()[a_winner[0][0]]
        else:
            # Random:
            #print('Explore action')
            move = random.choice(self.get_moves())

        # Save last_action: a value-index of the direction
        self.last_action = self.get_moves().index(move)

        return (move[0],move[1])

    def observation_step(self):
        if self.last_action is not None:
            # Process current experience state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices()
            self.ep_rew += self.last_reward


            if self.params['history']:
                self.last_4states = np.copy(self.still_4frames)

                self.still_4frames = np.dstack((self.still_4frames[:,:,4:],self.current_state))
                experience = (self.last_4states, float(self.last_reward), self.last_action, self.still_4frames, self.terminal)
            else:
            # Store last experience into memory
                experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Train
            self.train()

        # Next
        self.local_cnt += 1

        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt)/ float(self.params['eps_step']))

    def observationFunction(self):
        # Do observation
        self.terminal = False
        self.observation_step()

    # call after a round ends
    def final(self):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step()

        #log_file = open('results/dh_dqn_1600000_2.txt','a')
        #log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
        #                (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        #log_file.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.ai_won)))

        #sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f " %
        #                 (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps']))
        #sys.stdout.write("| Q: %10f | won: %r \n" % ((max(self.Q_global, default=float('nan')), self.ai_won)))
        #sys.stdout.flush()

    def train(self):
        # Train

        if (self.local_cnt > self.params['train_start']):

            batch = random.sample(self.replay_mem, self.params['batch_size'])

            batch_s = [] # States (s)
            batch_r = [] # Rewards (r)
            batch_a = [] # Actions (a)
            batch_n = [] # Next states (s')
            batch_t = [] # Terminal state (t)

            for i in batch:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])
            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)

            self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)


    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):
            actions_onehot[i][int(actions[i])] = 1
        return actions_onehot

    def getMatrix(self):
        # get prey's center
        for ent in self.entity.gamemap.entities:
            if ent.name == cfg.prey:
                center_x, center_y = ent.x, ent.y

        grid = np.ones((self.params['width'],self.params['width'],2))
        # range from center_x-r to center_x+r
        i = 0
        for x in range(center_x-cfg.radius, center_x+cfg.radius+1):
            j = 0
            for y in range(center_y-cfg.radius, center_y+cfg.radius+1):
                grid[i][j] = (x,y)
                j+=1
            i+=1
        return grid # a grid of coords

    def getStateMatrices(self):

        """ Return wall, ghosts, food, capsules matrices """
        def getWallMatrix():
            """ Return matrix with wall coordinates set to 1 """
            matrix = np.zeros((self.params['width'],self.params['height']), dtype=np.int8)
            full_grid = np.array(self.entity.gamemap.tiles["walkable"], dtype=np.int8)
            fov_grid_coords = self.getMatrix()
            i = 0
            for coord_row in fov_grid_coords:
                j = 0
                for coord in coord_row:
                    # Put cell vertically reversed in matrix
                    # check for out of bounds coords
                    if not self.engine.game_map.in_bounds(coord[0],coord[1]):
                        cell = 1

                    else:
                        cell = 0 if full_grid[int(coord[0])][int(coord[1])] else 1
                    matrix[-1-i][j] = cell
                    j+=1
                i+=1
            return matrix

        def getPreyMatrix():
            # Return matrix with pacman coordinates set to 1
            # find prey coords and replace fov coords == prey coords = 1
            # should be the center of fov
            matrix = np.zeros((self.params['width'],self.params['height']), dtype=np.int8)
            matrix[self.params['height']//2][self.params['width']//2] = 1
            return matrix

        def getMonsterMatrix():
            #Return matrix with ghost coordinates set to 1 """
            predator_coords = []
            for ent in self.entity.gamemap.entities:
                if ent.name == cfg.predator:
                    predator_coords.append([ent.x,ent.y])
            matrix = np.zeros((self.params['width'],self.params['height']), dtype=np.int8)
            #full_grid = np.array(self.entity.gamemap.tiles["walkable"], dtype=np.int8)
            fov_grid_coords = self.getMatrix()
            # find conjunction of fov_grid_coords and predator_coords
            i = 0

            for coord_row in fov_grid_coords:
                j = 0
                for coord in coord_row:
                    # if predator coord in fov coord then set to 1
                    if [coord[0],coord[1]] in predator_coords:
                        cell = 1
                        matrix[-1-i][j] = cell
                        i+=1

                j += 1
            return matrix


        def getHPMatrix():

            #Return matrix with food coordinates set to 1
            target_coords = []
            for ent in self.entity.gamemap.entities:
                if ent.name == cfg.target:
                    target_coords.append([ent.x,ent.y])
            matrix = np.zeros((self.params['width'],self.params['height']), dtype=np.int8)

            fov_grid_coords = self.getMatrix()
            # find conjunction of fov_grid_coords and predator_coords
            i = 0
            for coord_row in fov_grid_coords:
                j = 0
                for coord in coord_row:
                    # if predator coord in fov coord then set to 1
                    if [coord[0],coord[1]] in target_coords:
                        cell = 1
                        matrix[-1-i][j] = cell
                        j+=1
                i+=1
            return matrix

        # Create observation matrix as a combination of
        # wall, prey, monster and hp matrices
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((4, height, width))

        observation[0] = getWallMatrix()
        observation[1] = getPreyMatrix()
        observation[2] = getMonsterMatrix()
        observation[3] = getHPMatrix()
        observation = np.swapaxes(observation, 0, 2)

        return observation

    def registerInitialState(self): # inspects the starting state

        # Reset reward
        self.last_reward = 0
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices()
        self.last_4states = None
        self.still_4frames = np.tile(self.current_state,4)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.ai_won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.numeps += 1

    def getAction(self):
        move = self.getMove()

        return move

    def save_agent(self):
        # Save model
        # no of steps_no of rounds

        #self.qnet.save_ckpt('saves/dh_dqn_1600000_2_' + str(self.cnt) + '_' + str(self.numeps))
        print('Model saved')


    def perform(self) -> None:
        # initialise
        # perform action: initialise -> observe -> solicit action -> execute
        if self.new_round:
            self.registerInitialState() # initialise

        pickup = False
        beaten = False

        self.last_reward = cfg.MOVE_REWARD

        for ent in self.entity.gamemap.entities:
            if ent.name == cfg.predator:
                if (abs(self.entity.x-ent.x)==0 and abs(self.entity.y-ent.y)==1) or (abs(self.entity.x-ent.x)==1 and abs(self.entity.y-ent.y)==0):
                    self.last_reward = cfg.CAUGHT_BY_PLAYER
                    beaten = True
                    self.ai_won = False

                    break
            if ent.name == cfg.target and (self.entity.x == ent.x and self.entity.y == ent.y):

                self.last_reward = cfg.STEAL_POTION
                pickup = True

                self.score+=1
                break

        end = True
        print('AI hp: ',self.entity.fighter.hp)
        print('Your hp: ',self.engine.player.fighter.hp)
        # check if all hps are collected
        for i in self.entity.gamemap.entities:
            if i.name == cfg.target:
                end = False
        if end:
            c = 0
            for i in self.engine.player.inventory.items:
                if i.name == cfg.target:
                    c+=1
            if self.score > c:
                self.player_lose = True

            elif self.score < c:
                self.player_win = True

            else:
                if self.engine.player.fighter.hp > self.entity.fighter.hp:
                    self.player_win = True


        if pickup or beaten:
            # round ended
            self.new_round = True
            self.rounds += 1

            self.final()

            #f = open('rewards/dh_dqn_1600000_2.txt', 'a')
            #f.write(str(self.last_reward)+'\n')
            #f.close()

            if pickup:
                return PickupAction(self.entity).perform()

        else:
            self.new_round = False

            # choose a new action and execute it
        self.observationFunction()
        dir_x,dir_y = self.getAction()
        return MovementAction(
                self.entity,
                dir_x,dir_y,
            ).perform()

# moves away from the player
# moves towards the health potion
class DirectionalEnemy(BaseAI):
    def __init__(self, entity: Actor):
        super().__init__(entity)
        self.path: List[Tuple[int, int]] = []
        self.thiefScore = 0
        self.playerScore = 0
        self.reward = 0
        self.new_round = False
        self.player_lose = False
        self.player_win = False
        print('-------directional agent------')

    def won(self):
        return self.player_win

    def lost(self):
        return self.player_lose


    def is_new_round(self) -> None:
        return self.new_round

    def get_score(self) -> None:
        return self.thiefScore
    def get_rounds(self) -> None:
        return self.thiefScore+self.playerScore

    def perform(self) -> None:
        pickup = False
        beaten = False
        hide = False
        self.new_round = False
        self.reward += cfg.MOVE_REWARD
        predator_x,predator_y=0,0

        for target in self.entity.gamemap.entities:
            if self.engine.game_map.visible[target.x, target.y]:
                if target.name == cfg.predator:
                    predator_x,predator_y=target.x,target.y
                    hide = True
                    if (abs(self.entity.x-target.x)==1 and abs(self.entity.y-target.y)==0) or (abs(self.entity.x-target.x)==0 and abs(self.entity.y-target.y)==1):
                        self.playerScore += 1
                        self.reward += cfg.CAUGHT_BY_PLAYER
                        beaten = True
                        break
            # if there is an item, find the path to the item
            if target.name == cfg.target:
                    # check if the current position is an item
                if (self.entity.x, self.entity.y) == (target.x, target.y):
                        #print("Stealer picked up", target.name)
                    self.thiefScore += 1
                    pickup = True
                    self.reward += cfg.STEAL_POTION
                    break

                self.path = self.get_path_to(target.x, target.y)

        # priority 1: hide
        if hide:
            optimal_moves = []
            for pair in self.get_moves():
                # if it's further away from the predator, then add it
                if abs(self.entity.x+pair[0]-predator_x) > abs(self.entity.x-predator_x) or abs(self.entity.y+pair[1]-predator_y) > abs(self.entity.y-predator_y):
                    optimal_moves.append(pair)
            #print(optimal_moves)
            if optimal_moves:

                move = random.choice(optimal_moves)
            else:
                move = random.choice(self.get_moves())
            direction_x, direction_y = move[0], move[1]
        # priority 3: find path to the target if there is a path
        elif self.path:
            dest_x, dest_y = self.path.pop(0)
            direction_x, direction_y = dest_x - self.entity.x, dest_y - self.entity.y
        end = True
        print('AI hp: ',self.entity.fighter.hp)
        print('Your hp: ',self.engine.player.fighter.hp)
        # check if all hps are collected
        for i in self.entity.gamemap.entities:
            if i.name == cfg.target:
                end = False
        if end:
            c = 0
            for i in self.engine.player.inventory.items:
                if i.name == cfg.target:
                    c+=1
            if self.thiefScore > c:
                self.player_lose = True

            elif self.thiefScore < c:
                self.player_win = True

            else:
                if self.engine.player.fighter.hp > self.entity.fighter.hp:
                    self.player_win = True

        # priority 2: steals potion
        if pickup or beaten:
            self.new_round = True

            self.reward = 0

            if pickup:

                return PickupAction(self.entity).perform()

        return MovementAction(
            self.entity,
            direction_x, direction_y,
        ).perform()

class MockEnemy(BaseAI):
    def __init__(self, entity: Actor):
        super().__init__(entity)
        self.path: List[Tuple[int, int]] = []
        self.thiefScore = 0
        self.playerScore = 0
        self.reward = 0
        self.new_round = False
        self.player_lose = False
        self.player_win = False
        print('-------mock agent------')

    def won(self):
        return self.player_win

    def lost(self):
        return self.player_lose


    def is_new_round(self) -> None:
        return self.new_round

    def get_score(self) -> None:
        return self.thiefScore
    def get_rounds(self) -> None:
        return self.thiefScore+self.playerScore

    def perform(self) -> None:
        pickup = False
        beaten = False
        hide = False
        self.new_round = False
        self.reward += cfg.MOVE_REWARD
        predator_x,predator_y=0,0

        for target in self.entity.gamemap.entities:
            if self.engine.game_map.visible[target.x, target.y]:
                if target.name == cfg.predator:
                    predator_x,predator_y=target.x,target.y
                    hide = True
                    if (abs(self.entity.x-target.x)==1 and abs(self.entity.y-target.y)==0) or (abs(self.entity.x-target.x)==0 and abs(self.entity.y-target.y)==1):
                        self.playerScore += 1
                        self.reward += cfg.CAUGHT_BY_PLAYER
                        beaten = True
                        break
            # if there is an item, find the path to the item
            if target.name == cfg.target:
                    # check if the current position is an item
                if (self.entity.x, self.entity.y) == (target.x, target.y):
                        #print("Stealer picked up", target.name)
                    self.thiefScore += 1
                    pickup = True
                    self.reward += cfg.STEAL_POTION
                    break

                self.path = self.get_path_to(target.x, target.y)

        # priority 1: hide
        if random.uniform(0,1) <= 0.3:
            #print('random action')
            move = random.choice(self.get_moves())
            direction_x, direction_y = move[0], move[1]
        else:
            if hide:
                optimal_moves = []
                for pair in self.get_moves():
                    # if it's further away from the predator, then add it
                    if abs(self.entity.x+pair[0]-predator_x) > abs(self.entity.x-predator_x) or abs(self.entity.y+pair[1]-predator_y) > abs(self.entity.y-predator_y):
                        optimal_moves.append(pair)
                #print(optimal_moves)
                if optimal_moves:

                    move = random.choice(optimal_moves)
                else:
                    move = random.choice(self.get_moves())
                direction_x, direction_y = move[0], move[1]
            # priority 3: find path to the target if there is a path
            elif self.path:
                dest_x, dest_y = self.path.pop(0)
                direction_x, direction_y = dest_x - self.entity.x, dest_y - self.entity.y
        end = True
        print('AI hp: ',self.entity.fighter.hp)
        print('Your hp: ',self.engine.player.fighter.hp)
        # check if all hps are collected
        for i in self.entity.gamemap.entities:
            if i.name == cfg.target:
                end = False
        if end:
            c = 0
            for i in self.engine.player.inventory.items:
                if i.name == cfg.target:
                    c+=1
            if self.thiefScore > c:
                self.player_lose = True

            elif self.thiefScore < c:
                self.player_win = True

            else:
                if self.engine.player.fighter.hp > self.entity.fighter.hp:
                    self.player_win = True

        # priority 2: steals potion
        if pickup or beaten:
            self.new_round = True

            self.reward = 0

            if pickup:

                return PickupAction(self.entity).perform()

        return MovementAction(
            self.entity,
            direction_x, direction_y,
        ).perform()
