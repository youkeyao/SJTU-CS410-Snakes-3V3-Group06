import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(base_dir))
from env.chooseenv import make

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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

        state_batches = torch.Tensor(
            np.array([_[0] for _ in sample_batch])).to(DEVICE)
        action_batches = torch.LongTensor(
            np.array([_[1] for _ in sample_batch])).reshape(self.batch_size, 1).to(DEVICE)
        reward_batches = torch.Tensor(np.array([_[2] for _ in sample_batch])).reshape(
            self.batch_size, 1).to(DEVICE)
        next_state_batches = torch.Tensor(
            np.array([_[3] for _ in sample_batch])).to(DEVICE)
        done_batches = torch.Tensor(
            np.array([_[4] for _ in sample_batch])).to(DEVICE)

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

        Q = value + advantage - advantage.mean(-1).view(-1, 1)
        return Q


class DQN(object):
    def __init__(self, obs_dim, act_dim, args):
        self.path = os.path.dirname(os.path.abspath(__file__)) + "/trained_model"
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
            self.learn_step_counter = 0
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        b_s, b_a, b_r, b_s_, b_d = self.replay_buffer.get_batches()

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
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

# count reward
def get_reward(info, snake_index, reward, score):
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position'], dtype=object)
    snake_heads = [snake[0] for snake in snakes_position]
    step_reward = np.zeros(len(snake_index))
    for i in snake_index:
        step_reward[i] += 50 if score > 0 else -50
        step_reward[i] += 20 if reward[i] > 0 else -10

        self_head = np.array(snake_heads[i])
        dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
        step_reward[i] -= min(dists)

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
    obs_dim = 3
    print(f'observation dimension: {obs_dim}')

    actions_space = env.joint_action_space

    file_path = "/agent/" + args.opponent + "/submission.py"
    import_path = '.'.join(file_path.split('/')[-3:])[:-3]
    import_name = "my_controller"
    import_cmd = "from %s import %s" % (import_path, import_name)
    print(import_cmd)
    exec(import_cmd, globals())

    writer = SummaryWriter(os.path.dirname(os.path.abspath(__file__)) + '/run')
    model = DQN(obs_dim, act_dim, args)

    episode = 0
    win = [0 for i in range(100)]

    if args.load_model != 0:
        episode = args.load_model
        model.load_model(args.load_model)

    while episode < args.max_episodes:

        state = env.reset()

        obs = get_observations(state[0], ctrl_agent_index, height, width)

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
            next_obs = get_observations(next_state[0], ctrl_agent_index, height, width)

            reward = np.array(reward)
            episode_reward += reward

            step_reward = get_reward(info, ctrl_agent_index, reward, np.sum(episode_reward[:3])-np.sum(episode_reward[3:]))

            for i in range(ctrl_agent_num):
                model.replay_buffer.push(obs[i], actions[i], step_reward[i], next_obs[i], done)

            model.learn()

            if args.episode_length <= step or done:
                reward_tag = 'reward'
                acr_tag = 'accuracy'
                writer.add_scalars(reward_tag, global_step=episode,
                                   tag_scalar_dict={'snake_1': episode_reward[0], 'snake_2': episode_reward[1],
                                                    'snake_3': episode_reward[2], 'total': np.sum(episode_reward[0:3])})

                win.pop(0)
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    win.append(1)
                else:
                    win.append(0)
                print('Ep: ', episode, '| Ep_r: ', episode_reward,
                      '| eps: ', model.eps, '| acr: ', np.array(win).sum() / 100)
                writer.add_scalars(acr_tag, global_step=episode,
                                   tag_scalar_dict={'win_rate': np.array(win).sum() / 100})

                if episode % 1000 == 0:
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
