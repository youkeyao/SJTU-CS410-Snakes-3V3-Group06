DEPTH = 3
BEAN = 10
BODY = -100

# Action
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

# evaluate the gamestate
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

    def value_iteration(self, epsilon=0.001):
        U1 = {s: 0 for s in self.states}
        while True:
            convergent = True
            for s in self.states:
                if self.reward[s]:
                    continue
                qs = [self.expected_utility(a, s, U1) for a in Action.actlist]
                v = max(qs)
                if -epsilon > v - U1[s] or v - U1[s] > epsilon:
                    convergent = False
                U1[s] = v
            if convergent:
                break
        return U1

    def best_policy(self):
        U = self.value_iteration()
        policy = {}
        for s in self.states:
            actions = Action.actlist
            qs = [self.expected_utility(a, s, U) for a in actions]
            bestq = max(qs)
            bestIndex = qs.index(bestq)
            policy[s] = actions[bestIndex]
        return policy

class GameState:
    obs = {}
    food = []
    is_end = False
    def __init__(self, observation):
        self.obs = observation.copy()
        self.food = observation[1].copy()

    def generateSuccessor(self, index, action):
        successor = GameState(self.obs)
        index += 2
        head = tuple(successor.obs[index][0])
        tar = list(Action.go(head, action, self.obs['board_height'], self.obs['board_width']))
        for i in range(1, 8):
            for cor in successor.obs[i]:
                if cor == tar:
                    successor.is_end = True

        successor.obs[index].insert(0, tar)
        successor.obs[index].pop()
        
        return successor

    def evaluationFunction(self):
        # ans = 0
        head = tuple(self.obs[self.obs['controlled_snake_index']][0])
        # value = SnakeMDP(self.obs, 0.8).value_iteration()
        # for a in Action.actlist:
        #     ans += value[Action.go(head, a, self.obs['board_height'], self.obs['board_width'])]
        return SnakeMDP(self.obs, 0.8).value_iteration()[head]

class MinimaxAgent:
    def __init__(self, obs):
        self.obs = obs

    def value(self, gameState, index, depth, a, b):
        index %= 6
        if index == 0:
            return self.maxValue(gameState, index, depth + 1, a, b)[0]
        elif index < 3:
            return self.maxValue(gameState, index, depth, a, b)[0]
        else:
            return self.minValue(gameState, index, depth, a, b)[0]

    def maxValue(self, gameState, index, depth, a, b):
        if gameState.is_end or depth >= DEPTH:
            return [gameState.evaluationFunction(), None]

        v = -10000
        ac = Action.actlist[0]
        for action in Action.actlist:
            next = gameState.generateSuccessor(index, action)
            value = self.value(next, index+1, depth, a, b)
            if value > v:
                v = value
                ac = action
            if v >= b:
                return [v, ac]
            a = max(a, v)
        return [v, ac]

    def minValue(self, gameState, index, depth, a, b):
        if gameState.is_end:
            return [gameState.evaluationFunction(), None]

        v = 10000
        ac = Action.actlist[0]
        for action in Action.actlist:
            next = gameState.generateSuccessor(index, action)
            value = self.value(next, index+1, depth, a, b)
            if value < v:
                v = value
                ac = action
            if v <= a:
                return [v, ac]
            b = min(b, v)
        return [v, ac]

    def get_action(self):
        return self.maxValue(GameState(self.obs), 0, 0, -10000, 10000)[1]

def my_controller(observation, action_space, is_act_continuous=False):
    agent_action = []
    gameState = GameState(observation)

    # agent_action.append(maxValue(gameState, 0, -10000, 10000)[1])
    # agent_action.append(Action.right)
    # print(SnakeMDP(observation, 0.8).value_iteration()[tuple(observation[observation['controlled_snake_index']][0])])
    # print(gameState.evaluationFunction())
    # v = -10000
    # ans = Action.bottom

    head = tuple(observation[observation['controlled_snake_index']][0])
    value = SnakeMDP(observation, 0.9).value_iteration()
    R = SnakeMDP(observation, 0.9).reward
    p = SnakeMDP(observation, 0.9).best_policy()
    print(head)
    for i in range(observation['board_height']):
        for j in range(observation['board_width']):
            print(round(R[(i, j)]), end='\t')
        print('')
    return [Action.mapAct[SnakeMDP(observation, 0.9).best_policy()[head]]]

    # ac = Action.mapAct[MinimaxAgent(observation).get_action()]
    # print(ac)
    # return [ac]