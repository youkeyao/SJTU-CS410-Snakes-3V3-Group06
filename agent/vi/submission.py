BEAN = 50
DANGER = -100
WEAK_DANGER = -90
TAIL = 100
ENOUGH_LEN = 25

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
        self.gamma = gamma
        head = obs[obs['controlled_snake_index']][0]
        
        # snake collision reward
        for i in range(2, 8):
            for cor in obs[i]:
                self.reward[tuple(cor)] = DANGER
                if cor == obs[i][0] and i != obs['controlled_snake_index']:
                    for ac in Action.actlist:
                        pos = Action.go(tuple(cor), ac, obs['board_height'], obs['board_width'])
                        if self.reward[pos] >= 0:
                            self.reward[pos] = WEAK_DANGER
        self.reward[tuple(head)] = 0
        # Bean reward
        for cor in obs[1]:
            if self.reward[tuple(cor)] < 0:
                continue
            count = 0
            for ac in Action.actlist:
                pos = Action.go(tuple(cor), ac, obs['board_height'], obs['board_width'])
                count += self.reward[pos]
            self.reward[tuple(cor)] = BEAN if count > 3*DANGER else DANGER
        # if long enough, follow tail
        length = len(obs[obs['controlled_snake_index']])
        if length >= ENOUGH_LEN:
            tail = obs[obs['controlled_snake_index']][0]
            self.reward[tuple(obs[obs['controlled_snake_index']][length-1])] = TAIL

        self.transitions = {}
        for s in self.states:
            self.transitions[s] = {}
            for a in Action.actlist:
                if self.reward[s] < WEAK_DANGER:
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

    def value_iteration(self, epsilon=0.01):
        U1 = {s: 0 for s in self.states}
        while True:
            convergent = True
            for s in self.states:
                qs = [self.expected_utility(a, s, U1) for a in Action.actlist]
                v = max(qs)
                if -epsilon > v - U1[s] or v - U1[s] > epsilon:
                    convergent = False
                U1[s] = v
            if convergent:
                break
        return [U1, self.best_policy(U1)]

    def best_policy(self, U):
        policy = {}
        for s in self.states:
            actions = Action.actlist
            qs = [self.expected_utility(a, s, U) for a in actions]
            bestq = max(qs)
            bestIndex = qs.index(bestq)
            policy[s] = actions[bestIndex]
        return policy

def can_follow_tail(obs):
    length = len(obs[obs['controlled_snake_index']])
    head = tuple(obs[obs['controlled_snake_index']][0])
    tail = tuple(obs[obs['controlled_snake_index']][length - 1])

    act = ((tail[0] - head[0] + obs['board_height'] % obs['board_height'], tail[1] - head[1] + obs['board_width'] % obs['board_width']))
    if length >= ENOUGH_LEN and act in Action.actlist:
        return (True, act)
    else:
        return (False,)

def my_controller(observation, action_space, is_act_continuous=False):
    follow_tail = can_follow_tail(observation)
    if follow_tail[0]:
        action = follow_tail[1]
        return [Action.mapAct[action]]

    head = tuple(observation[observation['controlled_snake_index']][0])
    p = SnakeMDP(observation, 0.9).value_iteration()[1]
    action = p[head]
    return [Action.mapAct[action]]