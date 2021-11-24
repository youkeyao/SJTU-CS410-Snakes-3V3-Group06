import random

BEAN = 10
BODY = -100

class Action:
    top = [1, 0, 0, 0]
    bottom = [0, 1, 0, 0]
    left = [0, 0, 1, 0]
    right = [0, 0, 0, 1]
    actlist = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    mapAct = {
        actlist[0]: top,
        actlist[1]: bottom,
        actlist[2]: left,
        actlist[3]: right
    }

    def go(state, action, board_height, board_width):
        if action == (-1, 0):
            return ((state[0]+board_height-1) % board_height, state[1])
        elif action == (1, 0):
            return ((state[0]+1) % board_height, state[1])
        elif action == (0, 1):
            return (state[0], (state[1]+1) % board_width)
        elif action == (0, -1):
            return (state[0], (state[1]+board_width-1) % board_width)

class SnakeMDP:
    def __init__(self, obs, gamma=1.):
        self.states = set()
        self.reward = {}
        for i in range(obs['board_height']):
            for j in range(obs['board_width']):
                self.states.add((i, j))
                self.reward[(i,j)] = 0
        for cor in obs[1]:
            self.reward[tuple(cor)] = 10
        for i in range(2, 8):
            for cor in obs[i]:
                self.reward[tuple(cor)] = BODY
        self.reward[tuple(obs[obs['controlled_snake_index']][0])] = 0

        self.gamma = gamma
        self.transitions = {}
        for s in self.states:
            self.transitions[s] = {}
            for a in Action.actlist:
                if self.reward[s]:
                    self.transitions[s][a] = [(0.0, s)]
                else:
                    self.transitions[s][a] = [(1.0, Action.go(s, a, obs['board_height'], obs['board_width']))]

    def R(self, state):
        return self.reward[state]

    def T(self, state, action):
        if not self.transitions:
            raise ValueError("Transition model is missing")
        else:
            return self.transitions[state][action]

    def expected_utility(self, a, s, U):
        q = 0
        for p in self.T(s, a):
            q += p[0] * (self.R(p[1]) + self.gamma * U[p[1]])
        return q

    def policy_iteration(self):
        U = {s: 0 for s in self.states}
        pi = {s: random.choice(Action.actlist) for s in self.states}

        while True:
            U = self.policy_evaluation(pi)
            convergent, pi = self.policy_improvement(pi, U)
            if convergent:
                break
        return [U, pi]

    def policy_evaluation(self, pi, iteration_num=50):
        U = {s: 0 for s in self.states}
        for i in range(iteration_num):
            for s in self.states:
                U[s] = self.expected_utility(pi[s], s, U)
        return U

    def policy_improvement(self, pi, U):
        pi_new = self.best_policy(U)
        convergent = pi_new == pi
        return [convergent, pi_new]

    def best_policy(self, U):
        policy = {}
        for s in self.states:
            actions = Action.actlist
            qs = [self.expected_utility(a, s, U) for a in actions]
            bestq = max(qs)
            bestIndex = qs.index(bestq)
            policy[s] = actions[bestIndex]
        return policy

def my_controller(observation, action_space, is_act_continuous=False):
    head = tuple(observation[observation['controlled_snake_index']][0])
    p = SnakeMDP(observation, 0.9).policy_iteration()[1]
    return [Action.mapAct[p[head]]]