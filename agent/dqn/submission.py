import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")
ENOUGH_LEN = 25

class Action:
    top = [1, 0, 0, 0]
    bottom = [0, 1, 0, 0]
    left = [0, 0, 1, 0]
    right = [0, 0, 0, 1]
    actlist = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    mapAct = {
        0: top,
        1: bottom,
        2: left,
        3: right
    }

class Net(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
           nn.Conv2d(in_channels=obs_dim, out_channels=16, kernel_size=5, stride=1, padding=2),
           nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
           nn.Conv2d(16, 16, 5, 1, 2),
           nn.ReLU(),
        )

        self.fc1 = nn.Linear(6400, 32)
        self.fc1.weight.data.normal_(0, 0.1)
        self.outA = nn.Linear(32, act_dim)
        self.outA.weight.data.normal_(0, 0.1)
        self.outV = nn.Linear(32, 1)
        self.outV.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)

        advantage = self.outA(x)
        value = self.outV(x)

        Q = value + advantage - advantage.mean(-1).view(-1,1)
        return Q


class Agent(object):
    def __init__(self, obs_dim, act_dim):

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.eval_net = Net(obs_dim, act_dim)

    def choose_action(self, x):
        x = torch.FloatTensor(x)

        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()
        
        return action

    def load_model(self, path):
        eval = torch.load(path)
        self.eval_net.load_state_dict(eval)

def get_observations(state, agents_index, height, width):
    observations = []
    for i in agents_index:
        sample = []
        head = state[i+2][0]
        sample.append(getArea(state[i+2], height, width, head, True))

        other_snake = np.zeros((width, width))
        for j in range(2, 8):
            other_snake += getArea(state[j], height, width, head, True)
        sample.append(other_snake)

        sample.append(getArea(state[1], height, width, head, False))
        observations.append(sample)
    return np.array(observations)

def countPos(head, p, width, height):
    tmp = [0, 0]
    tmp[0] = int(p[0] - head[0] + height - 1) % height 
    tmp[1] = int(p[1] - head[1] + width * 3 / 2) % width
    return tmp

def getArea(state, height, width, head, isSnake):
    areas = np.zeros((height, width))
    for j in range(len(state)):
        p = tuple(countPos(head, state[j], width, height))
        if isSnake and j == 0:
            areas[p] = 10
            areas[((p[0] + 1) % height, p[1])] = 1
            areas[((p[0] + height - 1) % height, p[1])] = 1
            areas[(p[0], (p[1] + 1) % width)] = 1
            areas[(p[0], (p[1] + width - 1) % width)] = 1
        else:
            areas[p] = 5
    return np.concatenate((areas, areas))

def can_follow_tail(obs):
    length = len(obs[obs['controlled_snake_index']])
    head = tuple(obs[obs['controlled_snake_index']][0])
    tail = tuple(obs[obs['controlled_snake_index']][length - 1])

    act = ((tail[0] - head[0] + obs['board_height'] % obs['board_height'], tail[1] - head[1] + obs['board_width'] % obs['board_width']))
    if length >= ENOUGH_LEN and act in Action.actlist:
        return (True, Action.actlist.index(act))
    else:
        return (False,)

def my_controller(observation, action_space, is_act_continuous=False):
    follow_tail = can_follow_tail(observation)
    if follow_tail[0]:
        action = follow_tail[1]
        return [Action.mapAct[action]]

    board_width = observation['board_width']
    board_height = observation['board_height']
    o_index = observation['controlled_snake_index']  # 2, 3, 4, 5, 6, 7 -> indexs = [0,1,2,3,4,5]
    o_indexs_min = 3 if o_index > 4 else 0
    indexs = [o_indexs_min, o_indexs_min+1, o_indexs_min+2]
    obs = get_observations(observation, indexs, board_height, board_width)
    # agent
    agent = Agent(3, 4)
    eval = os.path.dirname(os.path.abspath(__file__)) + "/trained_model/eval_3000.pth"
    agent.load_model(eval)
    action = agent.choose_action(obs)[o_index-o_indexs_min-2]
    return [Action.mapAct[action]]