import math
import random
import numpy as np
import gomoku as gm
from gomoku import GomokuState

class UCTADPNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_reward = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.valid_actions())

def tree_policy(node, max_depth):
    while not node.state.is_game_over() and node.visits < max_depth:
        if not node.is_fully_expanded():
            return expand(node)
        else:
            node = best_child(node)
    return node

"""def expand(node):
    actions = node.state.valid_actions()
    untried_actions = [action for action in actions if action not in [child.action for child in node.children]]

    if untried_actions:
        action = random.choice(untried_actions)
        child_state = node.state.perform(action)
        child_node = UCTADPNode(child_state, parent=node, action=action)
        node.children.append(child_node)
        return child_node
    else:
        # Fully expanded, select the best child
        return best_child(node)
"""

def expand(node):
    actions = node.state.valid_actions()

    # Check for opponent's winning moves and prioritize blocking them
    blocking_moves = [action for action in actions if node.state.perform(action).current_score() == -1]

    if blocking_moves:
        action = random.choice(blocking_moves)
    else:
        # No opponent's winning moves, choose a random untried action
        untried_actions = [action for action in actions if action not in [child.action for child in node.children]]
        action = random.choice(untried_actions) if untried_actions else random.choice(actions)

    child_state = node.state.perform(action)
    child_node = UCTADPNode(child_state, parent=node, action=action)
    node.children.append(child_node)
    return child_node

# helper to find empty position in pth win pattern starting from (r,c)
def find_empty(state, p, r, c):
    if p == 0: # horizontal
        return r, c + state.board[gm.EMPTY, r, c:c+state.win_size].argmax()
    if p == 1: # vertical
        return r + state.board[gm.EMPTY, r:r+state.win_size, c].argmax(), c
    if p == 2: # diagonal
        rng = np.arange(state.win_size)
        offset = state.board[gm.EMPTY, r + rng, c + rng].argmax()
        return r + offset, c + offset
    if p == 3: # antidiagonal
        rng = np.arange(state.win_size)
        offset = state.board[gm.EMPTY, r - rng, c + rng].argmax()
        return r - offset, c + offset
    # None indicates no empty found
    return None

def look_ahead(state):

    # if current player has a win pattern with all their marks except one empty, they can win next turn
    player = state.current_player()
    sign = +1 if player == gm.MAX else -1
    magnitude = state.board[gm.EMPTY].sum() # no +1 since win comes after turn

    # check if current player is one move away to a win
    corr = state.corr
    idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, player] == state.win_size-1))
    if idx.shape[0] > 0:
        # find empty position they can fill to win, it is an optimal action
        p, r, c = idx[0]
        action = find_empty(state, p, r, c)
        return sign * magnitude, action

    # else, if opponent has at least two such moves with different empty positions, they can win in two turns
    opponent = gm.MIN if state.is_max_turn() else gm.MAX
    loss_empties = set() # make sure the 2+ empty positions are distinct
    idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, opponent] == state.win_size-1))
    for p, r, c in idx:
        pos = find_empty(state, p, r, c)
        loss_empties.add(pos)        
        if len(loss_empties) > 1: # just found a second empty
            score = -sign * (magnitude - 1) # opponent wins an extra turn later
            return score, pos # block one of their wins with next action even if futile

    # return 0 to signify no conclusive look-aheads
    return 0, None


def best_child(node, exploration_weight=1.0):
    children_with_rewards = [(child, child.total_reward / child.visits + exploration_weight * math.sqrt(2 * math.log(node.visits) / child.visits)) for child in node.children]
    return max(children_with_rewards, key=lambda x: x[1])[0]

def back_update(node, reward):
    while node is not None:
        node.visits += 1
        node.total_reward += reward
        reward = -reward  # Alternate rewards for players
        node = node.parent

def uct_adp(root_state, max_iterations=1000, max_depth=10):
    root_node = UCTADPNode(root_state)

    for _ in range(max_iterations):
        selected_node = tree_policy(root_node, max_depth)
        reward = evaluate(selected_node.state)
        back_update(selected_node, reward)

    # Choose the action with the highest value of UCT-ADP Progressive Bias
    bestChild = best_child(root_node, exploration_weight=0.0)  # Set exploration weight to 0 for pure exploitation
    return bestChild.action

def evaluate(state):
    # You may want to implement your own evaluation function based on the game state
    #return state.current_score()
    if state.is_game_over():
            winner = state.current_score()
            if winner == float('inf'):
                return 1.0  # Max player wins
            elif winner == float('-inf'):
                return -1.0  # Min player wins
            else:
                return 0.0  # It's a tie
    return 0.5


class Submission:
    def __init__(self, board_size, win_size):
        self.board_size = board_size
        self.win_size = win_size

    def __call__(self, state):
        return uct_adp(state)
    
if __name__ == "__main__":
    board_size = 5
    win_size = 3
    initial_state = GomokuState.blank(board_size, win_size)

    best_action = uct_adp(initial_state)
    print("Best Action:", best_action)




