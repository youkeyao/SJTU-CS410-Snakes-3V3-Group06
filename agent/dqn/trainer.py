import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from env.chooseenv import make

DEVICE = torch.device("cpu")
HEAD = -10
BODY = -5
TEAM_HEAD = -3
TEAM_BODY = -1
OPPONENT_HEAD = -4
OPPONENT_BODY = -2
BEAN = 10

# Memory for DQN
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.replay_buffer = []
        self.max_size = buffer_size
        self.batch_size = batch_size

    def push(self, state, logits, reward, next_state, done):
        transition_tuple = (state, logits, reward, next_state, done)
        if len(self.replay_buffer) >= self.max_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(transition_tuple)

    def get_batches(self):
        sample_batch = random.sample(self.replay_buffer, self.batch_size)

        state_batches = torch.Tensor(np.array([_[0] for _ in sample_batch])).to(DEVICE)
        action_batches = torch.LongTensor(np.array([_[1] for _ in sample_batch])).reshape(self.batch_size, 1).to(DEVICE)
        reward_batches = torch.Tensor(np.array([_[2] for _ in sample_batch])).reshape(self.batch_size, 1).to(DEVICE)
        next_state_batches = torch.Tensor(np.array([_[3] for _ in sample_batch])).to(DEVICE)
        done_batches = torch.Tensor(np.array([_[4] for _ in sample_batch])).to(DEVICE)

        return state_batches, action_batches, reward_batches, next_state_batches, done_batches

    def __len__(self):
        return len(self.replay_buffer)

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

        self.fc1 = nn.Linear(6400, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(64, act_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        out = self.out(x)
        return out


class DQN(object):
    def __init__(self, obs_dim, act_dim, args):
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.eps = args.epsilon
        self.gamma = args.gamma
        self.decay_speed = args.epsilon_speed
        self.batch_size = args.batch_size
        self.target_replace_iter = args.target_replace_iter

        self.learn_step_counter = 0

        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)
        self.eval_net, self.target_net = Net(obs_dim, act_dim).to(DEVICE), Net(obs_dim, act_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, evaluation=False):
        p = np.random.random()
        if p > self.eps or evaluation:
            x = torch.Tensor(x).to(DEVICE)
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
        else:
            action = np.random.randint(0, 4, (x.shape[0]))
        
        self.eps *= self.decay_speed
        return action

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # target parameter update
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        b_s, b_a, b_r, b_s_, b_d = self.replay_buffer.get_batches()

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load_model(self, episode):
        model_eval_path = self.path + "/eval_" + str(episode) + ".pth"
        model_target_path = self.path + "/target_" + str(episode) + ".pth"

        eval = torch.load(model_eval_path)
        target = torch.load(model_target_path)
        self.eval_net.load_state_dict(eval)
        self.target_net.load_state_dict(target)

    def save_model(self, episode):
        model_eval_path = self.path + "/eval_" + str(episode) + ".pth"
        torch.save(self.eval_net.state_dict(), model_eval_path)

        model_target_path = self.path + "/target_" + str(episode) + ".pth"
        torch.save(self.target_net.state_dict(), model_target_path)

# network input
def get_observations(state, agents_index, height, width):
    observations = []
    for i in agents_index:
        sample = []
        head = state[i+2][0]
        sample.append(getArea(state[i+2], height, width, head))
        for j in agents_index:
            if j != i:
                sample.append(getArea(state[j+2], height, width, head))
        for j in range(1, 8):
            if j != i+2 and j-2 not in agents_index:
                sample.append(getArea(state[j], height, width, head))
        observations.append(sample)
    return np.array(observations)
def countPos(head, p, width, height):
    tmp = [0, 0]
    tmp[0] = int(p[0] - head[0] + height - 1) % height 
    tmp[1] = int(p[1] - head[1] + width * 3 / 2) % width
    return tmp
def getArea(state, height, width, head):
    areas = np.zeros((height, width))
    for j in range(len(state)):
        p = tuple(countPos(head, state[j], width, height))
        if j == 0:
            areas[p] = 2
        else:
            areas[p] = 1
    return np.concatenate((areas, areas))
# def get_observations(state, agents_index, height, width):
#     observations = np.zeros((len(agents_index), width * height))
#     for i in agents_index:
#         areas = np.zeros((height, width))
#         head = state[i+2][0]
#         for j in range(1, 8):
#             for k in range(len(state[j])):
#                 p = tuple(countPos(head, state[j][k], width, height))
#                 if j == 1:
#                     areas[p] = BEAN
#                 elif j < 5:
#                     if k == 0:
#                         areas[p] = TEAM_HEAD
#                     else:
#                         areas[p] = TEAM_BODY
#                 else:
#                     if k == 0:
#                         areas[p] = OPPONENT_HEAD
#                     else:
#                         areas[p] = OPPONENT_BODY
#         observations[i][:] = areas.flatten()
#     return observations

# count reward
def get_reward(info, snake_index, reward, score):
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position'], dtype=object)
    snake_heads = [snake[0] for snake in snakes_position]
    step_reward = np.zeros(len(snake_index))
    for i in snake_index:
        if score == 1:
            step_reward[i] += 50
        elif score == 2:
            step_reward[i] -= 25
        elif score == 3:
            step_reward[i] += 10
        elif score == 4:
            step_reward[i] -= 5

        if reward[i] > 0:
            step_reward[i] += 20
        else:
            self_head = np.array(snake_heads[i])
            dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
            step_reward[i] -= min(dists)
            if reward[i] < 0:
                step_reward[i] -= 10

    return step_reward

def main(args):
    env = make(args.game_name, conf=None)

    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')
    ctrl_agent_index = [0, 1, 2]
    print(f'Agent control by the actor: {ctrl_agent_index}')
    ctrl_agent_num = len(ctrl_agent_index)

    width = env.board_width
    print(f'Game board width: {width}')
    height = env.board_height
    print(f'Game board height: {height}')

    act_dim = env.get_action_dim()
    print(f'action dimension: {act_dim}')
    obs_dim = 7
    print(f'observation dimension: {obs_dim}')

    actions_space = env.joint_action_space

    file_path = os.path.dirname(os.path.abspath(__file__)) + "/agent/" + args.opponent + "/submission.py"
    import_path = '.'.join(file_path.split('/')[-3:])[:-3]
    import_name = "my_controller"
    import_s = "from %s import %s" % (import_path, import_name)
    print(import_s)
    exec(import_s, globals())

    model = DQN(obs_dim, act_dim, args)

    episode = 0
    win = [0 for i in range(100)]

    if args.load_model != 0:
        episode = args.load_model
        model.load_model(args.load_model)

    while episode < args.max_episodes:

        state = env.reset()

        state_to_training = state[0]
        obs = get_observations(state_to_training, ctrl_agent_index, height, width)

        episode += 1
        step = 0
        episode_reward = np.zeros(6)

        while True:
            actions = model.choose_action(obs)
            team_actions = actions.tolist()
            for i in range(len(team_actions)):
                if team_actions[i] == 0:
                    team_actions[i] = [[1, 0, 0, 0]]
                elif team_actions[i] == 1:
                    team_actions[i] = [[0, 1, 0, 0]]
                elif team_actions[i] == 2:
                    team_actions[i] = [[0, 0, 1, 0]]
                elif team_actions[i] == 3:
                    team_actions[i] = [[0, 0, 0, 1]]
            opponent_actions = []
            for i in [5,6,7]:
                each = eval(import_name)(state[i-2], actions_space[0], False)
                opponent_actions.append(each)
            
            next_state, reward, done, _, info = env.step(team_actions+opponent_actions)
            next_state_to_training = next_state[0]
            next_obs = get_observations(next_state_to_training, ctrl_agent_index, height, width)

            reward = np.array(reward)
            episode_reward += reward

            if done:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=1)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=2)
                else:
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=0)
            else:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=3)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=4)
                else:
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=0)

            done = np.array([done] * ctrl_agent_num)

            for i in range(ctrl_agent_num):
                model.replay_buffer.push(obs[i], actions[i], step_reward[i], next_obs[i], done[i])

            model.learn()

            if args.episode_length <= step or (True in done):
                win.pop(0)
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    win.append(1)
                else:
                    win.append(0)
                print('Ep: ', episode, '| Ep_r: ', episode_reward, '| acr: ', np.array(win).sum()/100)
                if episode % 5000 == 0:
                    model.save_model(episode)
                env.reset()
                break

            obs = next_obs
            state = next_state
            step += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--max_episodes', default=50000, type=int)
    parser.add_argument('--episode_length', default=200, type=int)

    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epsilon', default=0.5, type=float)
    parser.add_argument('--epsilon_speed', default=0.99998, type=float)
    parser.add_argument('--target_replace_iter', default=100, type=int)
    parser.add_argument('--buffer_size', default=20000, type=int)

    parser.add_argument('--load_model', default=0, type=int)
    parser.add_argument('--opponent', default='random', type=str)

    args = parser.parse_args()
    main(args)
