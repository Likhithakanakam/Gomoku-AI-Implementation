"""
Do not change this file, it will be replaced by the instructor's copy
"""
import numpy as np
import gomoku as gm

EMPTY = 0
MIN = 1
MAX = 2
OrderedMoves = []
WIN_SCORE_CUTOFF = 10000

def outOfRange(state, move):
        return (move[0] < 0) or (move[0] >= state.board.shape[1]) or (move[1] < 0) or (move[1] >= state.board.shape[1])

def getScore(state, current_player, score = [0, 0, 2, 9, 15, 10000000]):
    # Purely offensive strategy
    totalScore = 0
    alreadyExistingHeadTails = set()
    #valid_actions = state.valid_actions()  # Assuming valid_actions gives the ordered moves
    opponent_player = 3 - current_player
    for move in OrderedMoves:
        # Check if the move is made by the current player
        if state.board[current_player, move[0], move[1]] == 1:
            boundaryList = [(1, 0), (0, 1), (1, 1), (-1, 1)]
            for vector in boundaryList:
                head = (move[0] + vector[0], move[1] + vector[1])
                tail = (move[0] - vector[0], move[1] - vector[1])
                curLen = 1
                headBlock = False
                tailBlock = False
                
                # Extending heads
                while not outOfRange(state, head) and state.board[current_player, head[0], head[1]] == 1:
                    curLen += 1
                    head = (head[0] + vector[0], head[1] + vector[1])
                headBlock = outOfRange(state, head) or state.board[opponent_player, head[0], head[1]] == 1

                # Extending tails
                while not outOfRange(state, tail) and state.board[current_player, tail[0], tail[1]] == 1:
                    curLen += 1
                    tail = (tail[0] - vector[0], tail[1] - vector[1])
                tailBlock = outOfRange(state, tail) or state.board[opponent_player, tail[0], tail[1]] == 1
                
                headTail = (head, tail)
                if headTail not in alreadyExistingHeadTails:
                    alreadyExistingHeadTails.add(headTail)
                    if curLen >= 5:
                        return score[5]  # Win condition
                    if (headBlock and not tailBlock) or (not headBlock and tailBlock):
                        if state.is_max_turn() and curLen == 4:  # A forced win
                            totalScore += 10000
                        else:
                            totalScore += score[curLen]  # score array should be defined
                    if not (headBlock or tailBlock):
                        if curLen == 4:
                            totalScore += 1000
                        totalScore += 2*score[curLen]  # score array should be defined

    if(totalScore != 0):
        return totalScore

    return totalScore

def evaluate(state, player):
    # Placeholder to be replaced with actual score array or heuristic logic
    return getScore(state, player)

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

def pre_select_moves(state):
    """
    Pre-select moves that are more likely to be beneficial.
    """
    valid_moves = state.valid_actions()
    # Apply some criteria to filter valid moves
    # Example: Select moves near existing pieces
    pre_selected_moves = []
    for move in valid_moves:
        if is_near_existing_piece(state, move):
            pre_selected_moves.append(move)
    return pre_selected_moves

def is_near_existing_piece(state, move):
    """
    Check if the move is near existing pieces on the board.
    """
    row, col = move
    board_size = state.board.shape[1]
    # Define proximity range (e.g., 1 cell around the move)
    for r in range(max(0, row-1), min(board_size, row+2)):
        for c in range(max(0, col-1), min(board_size, col+2)):
            if state.board[MIN, r, c] == 1 or state.board[MAX, r, c] == 1:
                return True
    return False

def alphaBeta(state, depth, alpha=-np.inf, beta=np.inf):

    score, action = look_ahead(state)
    if score != 0: return score, action

    player = state.current_player()
    opponent_player = 3 - player
    levelScore = evaluate(state, player) - evaluate(state, opponent_player)  # Assuming evaluate() returns a heuristic score of the current state

    if depth == 1 or (abs(levelScore) > WIN_SCORE_CUTOFF):
        return levelScore, None

    validMoves = pre_select_moves(state)#state.valid_actions()
    if state.is_max_turn():
        maxScore = -np.inf
        for move in validMoves:
            nextState = state.perform(move)
            OrderedMoves.append(move)
            tmpScore, action = alphaBeta(nextState, depth - 1, alpha, beta)
            maxScore = max(maxScore, tmpScore)
            alpha = max(alpha, maxScore)
            OrderedMoves.pop()
            if beta <= alpha:
                break
        return maxScore, None
    else:
        minScore = np.inf
        for move in validMoves:
            nextState = state.perform(move)
            OrderedMoves.append(move)
            tmpScore, action = alphaBeta(nextState, depth - 1, alpha, beta)
            minScore = min(minScore, tmpScore)
            beta = min(beta, minScore)
            OrderedMoves.pop()
            if beta <= alpha:
                break
        return minScore, None

def check_immediate_threat(state):
    score, action = look_ahead(state)

    if score != 0:
        # If score is not zero, it means there's either a winning move for the current player or a move to block the opponent's win.
        return True, action

    # No immediate threat or winning opportunity found
    return False, None

# Adaptation of the minimax function
def minimax(state, depth):

    bestMove = None
    maxScore = -np.inf
    allValidMoves = pre_select_moves(state)

    for move in allValidMoves:
        nextState = state.perform(move)
        OrderedMoves.append(move)
        
        curScore, blockMove = alphaBeta(nextState, depth)
        if blockMove is not None:
            return blockMove
        OrderedMoves.pop()
        if curScore > maxScore:
            maxScore = curScore
            bestMove = move

    return bestMove

# Policy wrapper
class SubmissionGitMinimax:
    def __init__(self, board_size, win_size, max_depth=2):
        self.max_depth = max_depth

    def __call__(self, state):
        action = minimax(state, self.max_depth)
        return action


