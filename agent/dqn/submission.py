import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

TEAM_HEAD = -3
TEAM_BODY = -1
OPPONENT_HEAD = -4
OPPONENT_BODY = -2
BEAN = 1

class Action:
    top = [1, 0, 0, 0]
    bottom = [0, 1, 0, 0]
    left = [0, 0, 1, 0]
    right = [0, 0, 0, 1]
    mapAct = {
        0: top,
        1: bottom,
        2: left,
        3: right
    }

class Net(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, act_dim)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


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

# network input
def get_observations(state, agents_index, height, width):
    observations = np.zeros((len(agents_index), 2 + width * height))
    areas = np.zeros((height, width))
    for i in range(1, 8):
        for j in range(len(state[i])):
            if i == 1:
                areas[tuple(state[i][j])] = BEAN
            elif i < 5:
                if j == 0:
                    areas[tuple(state[i][j])] = TEAM_HEAD
                else:
                    areas[tuple(state[i][j])] = TEAM_BODY
            else:
                if j == 0:
                    areas[tuple(state[i][j])] = OPPONENT_HEAD
                else:
                    areas[tuple(state[i][j])] = OPPONENT_BODY
    for i in agents_index:
        observations[i][:2] = state[i+2][0][:]
        observations[i][2:] = areas.flatten()
    return observations

def my_controller(observation, action_space, is_act_continuous=False):
    board_width = observation['board_width']
    board_height = observation['board_height']
    o_index = observation['controlled_snake_index']  # 2, 3, 4, 5, 6, 7 -> indexs = [0,1,2,3,4,5]
    o_indexs_min = 3 if o_index > 4 else 0
    indexs = [o_indexs_min, o_indexs_min+1, o_indexs_min+2]
    obs = get_observations(observation, indexs, board_height, board_width)
    # agent
    agent = Agent(202, 4)
    eval = os.path.dirname(os.path.abspath(__file__)) + "/eval_20000.pth"
    agent.load_model(eval)
    action = agent.choose_action(obs)[o_index-o_indexs_min-2]
    return [Action.mapAct[action]]