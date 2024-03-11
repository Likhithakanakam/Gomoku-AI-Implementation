import itertools as it
import random
import numpy as np
from scipy.signal import correlate
import gomoku as gm

EMPTY = 0
MIN = 1
MAX = 2

# helper function to get minimal path length to a game over state
# @profile
 
def outOfRange(state, move):
        return (move[0] < 0) or (move[0] >= state.board.shape[1]) or (move[1] < 0) or (move[1] >= state.board.shape[2])

def getScore(state):
    # Purely offensive strategy
    totalScore = 0
    alreadyExistingHeadTails = set()
    orderedMoves = list(zip(*np.nonzero(state.board == 1)))
     # Assuming valid_actions gives the ordered move
    state.orderedMoves = orderedMoves
   
    for move in state.orderedMoves:
        # Check if the move is made by the current player
        if state.board[state.current_player(), move[0], move[1]] == 1:
            boundaryList = [(1, 0), (0, 1), (1, 1), (-1, 1)]
            for vector in boundaryList:
                head = (move[0] + vector[0], move[1] + vector[1])
                tail = (move[0] - vector[0], move[1] - vector[1])
                curLen = 1
                headBlock = False
                tailBlock = False
                # Extending heads
                '''
                while not outOfRange(state,head) and state.board[state.current_player(), head[0], head[1]] == 1:
                    curLen += 1
                    head = (head[0] + vector[0], head[1] + vector[1])
                headBlock = outOfRange(state,head) or state.board[EMPTY, head[0], head[1]] == 0

                # Extending tails
                while not outOfRange(state,tail) and state.board[state.current_player(), tail[0], tail[1]] == 1:
                    curLen += 1
                    tail = (tail[0] - vector[0], tail[1] - vector[1])
                tailBlock = outOfRange(state,tail) or state.board[EMPTY, tail[0], tail[1]] == 0
                '''
                headTail = (head, tail)
                if headTail not in alreadyExistingHeadTails:
                    alreadyExistingHeadTails.add(headTail)
                    if curLen >= 5:
                        return float('inf')  # Win condition
                    if (headBlock and not tailBlock) or (not headBlock and tailBlock):
                        if state.is_max_turn() and curLen == 4:  # A forced win
                            totalScore += 10000
                        else:
                            totalScore += state.score[curLen]  # score array should be defined
                    if not (headBlock or tailBlock):
                        if curLen == 4:
                            totalScore += 1000
                        totalScore += 2*state.score[curLen]  # score array should be defined

    return totalScore

def evaluate(state):
    # Placeholder to be replaced with actual score array or heuristic logic
    state.score = [0,0, 2, 9, 15, float('inf')]
    #state.score = [0,1,10,100,1000,float('inf')]
    return getScore(state)

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

# fast look-aheads to short-circuit the minimax search when possible

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

# recursive minimax search with additional pruning
# @profile
def minimax(state, max_depth, alpha=-np.inf, beta=np.inf):

    # check fast look-ahead before trying minimax
    score, action = look_ahead(state)
    if score != 0: return score, action

    # check for game over base case with no valid actions
    if state.is_game_over():
        return state.current_score(), None

    # have to try minimax, prepare the valid actions
    # should be at least one valid action if this code is reached
    actions = state.valid_actions()

    # prioritize actions near non-empties but break ties randomly
    rank = -state.corr[:, 1:].sum(axis=(0,1)) - np.random.rand(*state.board.shape[1:])
    rank = rank[state.board[gm.EMPTY] > 0] # only empty positions are valid actions
    scrambler = np.argsort(rank)

    # check for max depth base case
    if max_depth == 0:
        return state.current_score(), actions[scrambler[0]]
    
    # custom pruning: stop search if no path from this state wins within max_depth turns
    if evaluate(state) > max_depth: return 0, actions[scrambler[0]]

    # alpha-beta pruning
    best_action = None
    if state.is_max_turn():
        bound = -np.inf
        for a in scrambler:
            action = actions[a]
            child = state.perform(action)
            utility, _ = minimax(child, max_depth-1, alpha, beta)

            if utility > bound: bound, best_action = utility, action
            if bound >= beta: break
            alpha = max(alpha, bound)

    else:
        bound = +np.inf
        for a in scrambler:
            action = actions[a]
            child = state.perform(action)
            utility, _ = minimax(child, max_depth-1, alpha, beta)

            if utility < bound: bound, best_action = utility, action
            if bound <= alpha: break
            beta = min(beta, bound)

    return bound, best_action

# Policy wrapper
class Submission:
    def __init__(self, board_size, win_size, max_depth=2):
        self.max_depth = max_depth

    def __call__(self, state):
        _, action = minimax(state, self.max_depth)
        return action


