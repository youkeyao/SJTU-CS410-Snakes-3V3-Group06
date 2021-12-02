DEPTH = 3

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

class GameState:
    obs = {}
    is_end = False
    def __init__(self, observation):
        self.obs = {
            1: observation[1].copy(),
            2: observation[2].copy(),
            3: observation[3].copy(),
            4: observation[4].copy(),
            5: observation[5].copy(),
            6: observation[6].copy(),
            7: observation[7].copy(),
            'board_width': observation['board_width'],
            'board_height': observation['board_height'],
        }

    def generateSuccessor(self, index, action):
        successor = GameState(self.obs)
        index += 2
        head = tuple(successor.obs[index][0])
        tar = list(Action.go(head, action, self.obs['board_height'], self.obs['board_width']))
        for i in range(1, 8):
            for cor in successor.obs[i]:
                if cor == tar:
                    successor.is_end = True
                    if i == 1:
                        successor.obs[index].append(successor.obs[index][-1])
                    else:
                        successor.obs[index].clear()

        successor.obs[index].insert(0, tar)
        successor.obs[index].pop()
        
        return successor

    def evaluationFunction(self):
        ans = 0
        for i in range(2, 8):
            if i < 5:
                ans += len(self.obs[i])
            else:
                ans -= len(self.obs[i])
        return ans

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

    def get_action(self, index):
        return self.maxValue(GameState(self.obs), index-2, 0, -10000, 10000)[1]

def my_controller(observation, action_space, is_act_continuous=False):
    ac = Action.mapAct[MinimaxAgent(observation).get_action(observation['controlled_snake_index'])]
    return [ac]