import math
import random
import numpy as np
from gomoku import GomokuState, MAX, MIN, EMPTY

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.total_reward = 0
        self.visits = 0
        self.untried_actions = state.valid_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

class UCT_ADP:
    def __init__(self, root_state):
        self.root = Node(root_state)

    def uct_search(self):
        iterations = (self.root.state.board.shape[1])**2

        """
        r = dict()
        for action in pre_select_moves(self.root.state):
            r[action] = self.evaluate(self.root
        """
        for _ in range(500):
            leaf = self.tree_policy(self.root)
            reward, block_move = self.evaluate_adp(leaf)
            if block_move is not None:
                return block_move
            self.backpropagate(leaf, reward)

        return (self.best_child(self.root)).action

    def tree_policy(self, node):
        while not node.state.is_game_over():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = self.best_child(node)
        return node

    def expand(self, node):
        """
        Expand the current node by adding a new child node.
        """
        # Step 1: Check for actions that can realize VCF or VCT strategy
        vcf_vct_actions = self.find_vcf_vct_actions(node.state)
        
        # Step 2: If no VCF/VCT actions, consider all untried actions
        if not vcf_vct_actions:
            actions = node.state.valid_actions()
            untried_actions = actions #pre_select_moves(node.state)
            actions_to_consider = untried_actions
        else:
            actions_to_consider = vcf_vct_actions

        # Step 3: Choose a random action from the actions to consider
        action = random.choice(actions_to_consider)
        #self.untried_actions.remove(action)

        # Step 4: Add a new child node with the selected action
        if action not in node.children:
            new_state = node.state.perform(action)
            child_node = Node(new_state, node, action)
            node.children[action] = child_node

        return node.children[action]

    def find_vcf_vct_actions(self, state):
        """
        Find actions that can realize VCF or VCT strategy.
        """
        vcf_vct_actions = []
        actions = state.valid_actions()
        untried_actions = actions #pre_select_moves(state)
        for action in untried_actions:
            # Check if the action can realize VCF or VCT
            # This requires game-specific logic to determine if an action leads to VCF or VCT
            if self.can_realize_vcf_vct(state, action):
                vcf_vct_actions.append(action)
        return vcf_vct_actions
    
    def can_realize_vcf_vct(self, state, action):
        """
        Check if an action can realize VCF or VCT.
        """
        # Apply the action to the state
        new_state = state.perform(action)
        player = state.current_player()

        # Check for VCF
        if self.is_victory_of_continuous_four(new_state, action, player):
            return True

        # Check for VCT
        if self.is_victory_of_continuous_three(new_state, action, player):
            return True

        return False
    
    def is_victory_of_continuous_four(self, state, action, player):
        """
        Check if the action creates a Victory of Continuous Four (VCF) situation.
        """
        row, col = action

        # Iterate over all directions
        for direction in range(4):  # horizontal, vertical, diagonal, anti-diagonal
            if state.corr[direction, player, row, col] == 4:
                # Check if the line of four is not blocked at either end
                if self.is_line_open(state, row, col, direction, 4, player):
                    return True

        return False

    def is_victory_of_continuous_three(self, state, action, player):
        """
        Check if the action creates a Victory of Continuous Three (VCT) situation.
        """
        row, col = action

        # Iterate over all directions
        for direction in range(4):  # horizontal, vertical, diagonal, anti-diagonal
            if state.corr[direction, player, row, col] == 3:
                # Check if the line of three has the potential to extend to four
                if self.is_potential_vct(state, row, col, direction):
                    return True

        return False
        
    def is_line_open(self, state, row, col, direction, length, player):
        """
        Check if a line of a specified length is open at both ends.
        """
        directions = [
        (1, 0),  # Horizontal (right)
        (0, 1),  # Vertical (down)
        (1, 1),  # Diagonal (down-right)
        (-1, 1)  # Anti-diagonal (up-right)
        ]
        dr, dc = directions[direction]

        # Check open end in the positive direction
        r, c = row + length * dr, col + length * dc
        if not self.is_valid_position(state, r, c) or state.board[player, r, c] == 1:
            return False

        # Check open end in the negative direction
        r, c = row - dr, col - dc
        if not self.is_valid_position(state, r, c) or state.board[player, r, c] == 1:
            return False

        return True
    
    def is_potential_vct(self, state, row, col, direction):
        """
        Check if a line of three has the potential to extend to four.
        """
        directions = [
        (1, 0),  # Horizontal (right)
        (0, 1),  # Vertical (down)
        (1, 1),  # Diagonal (down-right)
        (-1, 1)  # Anti-diagonal (up-right)
        ]
        dr, dc = directions[direction]

        # Check for a potential extension in the positive direction
        r, c = row + 3 * dr, col + 3 * dc
        if self.is_valid_position(state, r, c) and state.board[EMPTY, r, c] == 1:
            return True

        # Check for a potential extension in the negative direction
        r, c = row - dr, col - dc
        if self.is_valid_position(state, r, c) and state.board[EMPTY, r, c] == 1:
            return True

        return False

    def best_child(self, node, exploration_value=1.41):

        best_score = -float('inf')
        best_children = []
        for child in node.children.values():
            uct_value = (child.total_reward / child.visits +
                         exploration_value * math.sqrt(2 * math.log(node.visits) / child.visits))
            if uct_value > best_score:
                best_children = [child]
                best_score = uct_value
            elif uct_value == best_score:
                best_children.append(child)
        return random.choice(best_children)

    def exponential_heuristic(self, state):
        """
        Exponential heuristic function for board evaluation.
        """
        score = 0
        board_size = state.board.shape[1]
        # Define the exponential scoring parameters
        # Example: exponential scoring for continuous pieces
        for player in [MIN, MAX]:
            for direction in range(4):  # horizontal, vertical, diagonal, anti-diagonal
                for row in range(board_size):
                    for col in range(board_size):
                        continuous_count = state.corr[direction, player, row, col]
                        # Exponential scoring formula
                        score += (2 ** continuous_count) if player == MAX else -(2 ** continuous_count)
        return score
    
    def evaluate_adp(self, leaf):
        # Implementation of the ADP evaluation
        opponent_win_imminent, block_move = check_immediate_threat(leaf.state)
        if opponent_win_imminent:
            return -1000, block_move  # Assign a large negative value to indicate blocking is required
        
        return self.evaluateMinMax(leaf), None
    
    def outOfRange(self, move):
        return (move[0] < 0) or (move[0] >= self.board.shape[1]) or (move[1] < 0) or (move[1] >= self.board.shape[2])

    def getScore(self,state):
        # Purely offensive strategy
        totalScore = 0
        alreadyExistingHeadTails = set()
        self.orderedMoves = self.state.valid_actions()  # Assuming valid_actions gives the ordered moves
        for move in self.orderedMoves:
            # Check if the move is made by the current player
            if self.board[self.current_player(), move[0], move[1]] == 1:
                boundaryList = [(1, 0), (0, 1), (1, 1), (-1, 1)]
                for vector in boundaryList:
                    head = (move[0] + vector[0], move[1] + vector[1])
                    tail = (move[0] - vector[0], move[1] - vector[1])
                    curLen = 1
                    headBlock = False
                    tailBlock = False
                    
                    # Extending heads
                    while not self.outOfRange(head) and self.board[self.current_player(), head[0], head[1]] == 1:
                        curLen += 1
                        head = (head[0] + vector[0], head[1] + vector[1])
                    headBlock = self.outOfRange(head) or self.board[EMPTY, head[0], head[1]] == 0

                    # Extending tails
                    while not self.outOfRange(tail) and self.board[self.current_player(), tail[0], tail[1]] == 1:
                        curLen += 1
                        tail = (tail[0] - vector[0], tail[1] - vector[1])
                    tailBlock = self.outOfRange(tail) or self.board[EMPTY, tail[0], tail[1]] == 0
                    
                    headTail = (head, tail)
                    if headTail not in alreadyExistingHeadTails:
                        alreadyExistingHeadTails.add(headTail)
                        if curLen >= 5:
                            return float('inf')  # Win condition
                        if (headBlock and not tailBlock) or (not headBlock and tailBlock):
                            if self.is_max_turn() and curLen == 4:  # A forced win
                                totalScore += 10000
                            else:
                                totalScore += self.score[curLen]  # score array should be defined
                        if not (headBlock or tailBlock):
                            if curLen == 4:
                                totalScore += 1000
                            totalScore += 2*self.score[curLen]  # score array should be defined

        return totalScore

    def evaluateMinMax(self,state):
        # Placeholder to be replaced with actual score array or heuristic logic
        self.score = [0, 1, 10, 100, 1000, float('inf')]
        return self.getScore(state)
    
    def evaluate1(self, leaf):
        player = leaf.state.current_player()
        return self.confront_heuristic(leaf.state, leaf.action[0], leaf.action[1],player)

    def oppsite_who(self, wh):
        return 3 - wh

    def my_heuristic(self, state, x, y, who):
        heuristic = 0.0
        factor = 0.90
        MAX_BOARD = state.board.shape[1]
        win_size = 5
        def is_valid_position(row, col):
            return 0 <= row < MAX_BOARD and 0 <= col < MAX_BOARD

        def check_line(delta_x, delta_y):
            count = 0
            opponent_count = 0
            for step in range(1, win_size):
                new_x = x + step * delta_x
                new_y = y + step * delta_y
                if not is_valid_position(new_x, new_y):
                    opponent_count += 1
                    break
                if state.board[who, new_x, new_y]:
                    count += 1
                elif state.board[self.oppsite_who(who), new_x, new_y]:
                    opponent_count += 1
                    break
                else:
                    break
            return count, opponent_count
        
        # Check horizontal, vertical, and both diagonal lines
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count1, opponent1 = check_line(dx, dy)
            count2, opponent2 = check_line(-dx, -dy)
            total = count1 + count2
            opponents = opponent1 + opponent2

            if total >= win_size - 1:
                heuristic += 10 ** 5
            else:
                if opponents == 0 and total > 0:
                    heuristic += 10 ** (total + 1.0)
                elif opponents == 1 and total > 0:
                    heuristic += 10 ** total
                elif opponents == 2:
                    heuristic += 0

        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                
                # - OO O - or - O OO -
                if self.is_valid_position(state, x+3*i, y+3*j) and self.is_valid_position(state, x-2*i, y-2*j):
                    if state.board[who, x-1*i, y-1*j] == 1 and state.board[who, x-2*i, y-2*j] == 0 and state.board[who, x+1*i, y+1*j] == 0 and state.board[who, x+2*i, y+2*j] == 1 and state.board[who, x+3*i, y+3*j] == 0:
                        heuristic += 1000 * factor - 100
                if self.is_valid_position(state, x+4*i, y+4*j) and self.is_valid_position(state, x-1*i, y-1*j):
                    if state.board[who, x+1*i, y+1*j] == 0 and state.board[who, x+2*i, y+2*j] == 1 and state.board[who, x+3*i, y+3*j] == 1 and state.board[who, x-1*i, y-1*j] == 0 and state.board[who, x+4*i, y+4*j] == 0:
                        heuristic += 1000 * factor
                    # Pattern - 0O O -
                    elif state.board[who, x+1*i, y+1*j] == 1 and state.board[who, x+2*i, y+2*j] == 0 and state.board[who, x+3*i, y+3*j] == 1 and state.board[who, x-1*i, y-1*j] == 0 and state.board[who, x+4*i, y+4*j] == 0:
                        heuristic += 1000 * factor - 100
                # Pattern -XOO O - or -XO OO -
                if self.is_valid_position(state, x+3*i, y+3*j) and self.is_valid_position(state, x-2*i, y-2*j):
                    # Pattern -XO0 O -
                    if state.board[who, x+1*i, y+1*j] == 0 and state.board[who, x-1*i, y-1*j] == 1 and state.board[self.oppsite_who(who), x-2*i, y-2*j] == 1 and state.board[who, x+2*i, y+2*j] == 1 and state.board[who, x+3*i, y+3*j] == 0:
                        heuristic += 100 * factor * factor - 10
                # Check if the positions for the pattern are valid
                if self.is_valid_position(state, x+4*i, y+4*j) and self.is_valid_position(state, x-1*i, y-1*j):
                    # Pattern -X0O O -
                    if state.board[who, x+1*i, y+1*j] == 1 and state.board[who, x+2*i, y+2*j] == 0 and state.board[who, x+3*i, y+3*j] == 1 and state.board[self.oppsite_who(who), x-1*i, y-1*j] == 1 and state.board[who, x+4*i, y+4*j] == 0:
                        heuristic += 100 * factor * factor - 10
                    # Pattern -XOO 0 -
                    elif state.board[who, x+1*i, y+1*j] == 0 and state.board[who, x+2*i, y+2*j] == 1 and state.board[who, x+3*i, y+3*j] == 1 and state.board[who, x-1*i, y-1*j] == 0 and state.board[self.oppsite_who(who), x+4*i, y+4*j] == 1:
                        heuristic += 100 * factor * factor
                    # Pattern -X0 OO -
                    elif state.board[who, x+1*i, y+1*j] == 0 and state.board[who, x+2*i, y+2*j] == 1 and state.board[who, x+3*i, y+3*j] == 1 and state.board[self.oppsite_who(who), x-1*i, y-1*j] == 1 and state.board[who, x+4*i, y+4*j] == 0:
                        heuristic += 100 * factor * factor
                    # Pattern -XO O0 -
                    elif state.board[who, x+1*i, y+1*j] == 1 and state.board[who, x+2*i, y+2*j] == 0 and state.board[who, x+3*i, y+3*j] == 1 and state.board[who, x-1*i, y-1*j] == 0 and state.board[self.oppsite_who(who), x+4*i, y+4*j] == 1:
                        heuristic += 100 * factor * factor
                # Pattern -XO 0O -
                if self.is_valid_position(state, x+2*i, y+2*j) and self.is_valid_position(state, x-3*i, y-3*j):
                    if state.board[who, x+1*i, y+1*j] == 1 and state.board[who, x-1*i, y-1*j] == 0 and state.board[who, x-2*i, y-2*j] == 1 and state.board[who, x+2*i, y+2*j] == 0 and state.board[self.oppsite_who(who), x-3*i, y-3*j] == 1:
                        heuristic += 100 * factor * factor

                # Pattern -XOOO 0- and -XOO O0-
                if self.is_valid_position(state, x-5*i, y-5*j):
                    if state.board[who, x-1*i, y-1*j] == 0 and state.board[who, x-2*i, y-2*j] == 1 and state.board[who, x-3*i, y-3*j] == 1 and state.board[who, x-4*i, y-4*j] == 1 and state.board[self.oppsite_who(who), x-5*i, y-5*j] == 1:
                        heuristic += 1000 * factor * factor - 10
                    elif state.board[who, x-1*i, y-1*j] == 1 and state.board[who, x-2*i, y-2*j] == 0 and state.board[who, x-3*i, y-3*j] == 1 and state.board[who, x-4*i, y-4*j] == 1 and state.board[self.oppsite_who(who), x-5*i, y-5*j] == 1:
                        heuristic += 1000 * factor * factor - 100

                # Pattern -XOO 0O-
                if self.is_valid_position(state, x+1*i, y+1*j) and self.is_valid_position(state, x-4*i, y-4*j):
                    if state.board[who, x-1*i, y-1*j] == 0 and state.board[who, x-2*i, y-2*j] == 1 and state.board[who, x-3*i, y-3*j] == 1 and state.board[self.oppsite_who(who), x-4*i, y-4*j] == 1 and state.board[who, x+1*i, y+1*j] == 1:
                        heuristic += 1000 * factor * factor - 100

                # Check for pattern - O O O -
                if self.is_valid_position(state, x-3*i, y-3*j) and self.is_valid_position(state, x+3*i, y+3*j):
                    if state.board[who, x-3*i, y-3*j] == 0 and state.board[who, x-2*i, y-2*j] == 1 and state.board[who, x-1*i, y-1*j] == 0 and state.board[who, x+1*i, y+1*j] == 0 and state.board[who, x+2*i, y+2*j] == 1 and state.board[who, x+3*i, y+3*j] == 0:
                        heuristic += 100 * factor * factor * factor

                # Check for pattern - O O -
                if self.is_valid_position(state, x-1*i, y-1*j) and self.is_valid_position(state, x+3*i, y+3*j):
                    if state.board[who, x-1*i, y-1*j] == 0 and state.board[who, x+1*i, y+1*j] == 0 and state.board[who, x+2*i, y+2*j] == 1 and state.board[who, x+3*i, y+3*j] == 0:
                        heuristic += 10 * factor * factor

        # Check nearby positions for the current player's pieces and update the heuristic
        nearby_positions = [(1, 2), (-1, 2), (1, -2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]
        for dx, dy in nearby_positions:
            if self.is_valid_position(state, x + dx, y + dy) and state.board[who, x + dx, y + dy] == 1:
                heuristic += 10

        return heuristic

    def confront_heuristic(self, state, x, y, who): 
        """The function calculate the confront_heuristic after adding an adjacent point"""
        """delete each old heuristic value in 8 directions"""
        beta = 1/6
        
        return beta * self.my_heuristic(state, x, y, who) + (1-beta) * self.my_heuristic(state, x, y, self.oppsite_who(who))/10

    def evaluate(self, state):
        """
        Evaluate the board using the provided heuristic formula.
        """
        score = 0
        board_size = state.board.shape[1]  # Assuming square board

        # Define factors for open and half-closed lines
        factor_j = 0.90  # Adjust as needed
        factor_k = 0.90  # Adjust as needed

        for player in [MIN, MAX]:
            for row in range(board_size):
                for col in range(board_size):
                    if state.board[player, row, col] == 1:
                        open_length, hclose_length = self.line_lengths(state, row, col, player)
                        score += (10 ** open_length) * factor_j
                        if hclose_length > 0:
                            score += (10 ** (hclose_length - 1)) * factor_k

        return score + self.exponential_heuristic(state)

    def line_lengths(self, state, row, col, player):
        """
        Calculate the lengths of open and half-closed lines from a specific position.
        """
        open_length, hclose_length = 0, 0
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]  # Horizontal, Vertical, Diagonal, Anti-Diagonal

        for dr, dc in directions:
            temp_open, temp_hclose = self.check_line(state, row, col, dr, dc, player)
            open_length = max(open_length, temp_open)
            hclose_length = max(hclose_length, temp_hclose)

        return open_length, hclose_length

    def check_line(self, state, row, col, dr, dc, player):
        """
        Check a line in one direction and return lengths of open and half-closed lines.
        """
        count = 0
        r, c = row, col

        # Move in the specified direction
        while self.is_valid_position(state, r, c) and state.board[player, r, c] == 1:
            count += 1
            r += dr
            c += dc

        # Check if the line is open or half-closed
        if self.is_valid_position(state, r, c) and state.board[EMPTY, r, c] == 1:
            return count, 0  # Open line
        else:
            return 0, count  # Half-closed line

    def is_valid_position(self, state, row, col):

        # Check if a position is within the bounds of the board.

        board_size = state.board.shape[1]
        return 0 <= row < board_size and 0 <= col < board_size

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            reward = 1 - reward
            node = node.parent

def check_immediate_threat(state):
    score, action = look_ahead(state)

    if score != 0:
        # If score is not zero, it means there's either a winning move for the current player or a move to block the opponent's win.
        return True, action

    # No immediate threat or winning opportunity found
    return False, None

# helper to find empty position in pth win pattern starting from (r,c)
def find_empty(state, p, r, c):
    if p == 0: # horizontal
        return r, c + state.board[EMPTY, r, c:c+state.win_size].argmax()
    if p == 1: # vertical
        return r + state.board[EMPTY, r:r+state.win_size, c].argmax(), c
    if p == 2: # diagonal
        rng = np.arange(state.win_size)
        offset = state.board[EMPTY, r + rng, c + rng].argmax()
        return r + offset, c + offset
    if p == 3: # antidiagonal
        rng = np.arange(state.win_size)
        offset = state.board[EMPTY, r - rng, c + rng].argmax()
        return r - offset, c + offset
    # None indicates no empty found
    return None

def look_ahead(state):
    # if current player has a win pattern with all their marks except one empty, they can win next turn
    player = state.current_player()
    sign = +1 if player == MAX else -1
    magnitude = state.board[EMPTY].sum()  # no +1 since win comes after turn

    # check if current player is one move away to a win
    corr = state.corr
    idx = np.argwhere((corr[:, EMPTY] == 1) & (corr[:, player] == state.win_size - 1))
    if idx.shape[0] > 0:
        for p, r, c in idx:
            pos = find_empty(state, p, r, c)
            if pos is not None:
                score = sign * magnitude
                return score, pos

    # check if opponent has at least two moves with different empty positions, they can win in two turns
    opponent = MIN if state.is_max_turn() else MAX
    loss_empties = set()
    idx = np.argwhere((corr[:, EMPTY] == 1) & (corr[:, opponent] == state.win_size - 1))
    for p, r, c in idx:
        pos = find_empty(state, p, r, c)
        loss_empties.add(pos)
        if len(loss_empties) > 1:
            score = -sign * (magnitude - 1)
            return score, pos

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

class Submission:
    def __init__(self, board_size, win_size):
        self.board_size = board_size
        self.win_size = win_size
        # Initialize other necessary components for UCT-ADP

    def __call__(self, state):

        """
        # First, Check for immediate threats
        opponent_win_imminent, block_move = check_immediate_threat(state)
        if opponent_win_imminent:
            return block_move  
        """

        # Second, check for victory moves
        victory_move = self.victory_move_check(state)
        if victory_move is not None:
            return victory_move
        
        # If no immediate victory move, use UCT-ADP to find the best move
        best_action = self.uct_adp_search(state)  # Adjust iterations as needed
        return best_action

    def victory_move_check(self, state):
        """
        Check for moves that result in a victory of continuous four or three.
        """
        for move in pre_select_moves(state):
            hypothetical_state = state.perform(move)
            if self.check_continuous_four(hypothetical_state) or self.check_continuous_three(hypothetical_state):
                return move
        return None

    def check_continuous_four(self, state):
        """
        Check for four continuous pieces of the current player.
        """
        player = state.current_player()
        for direction in range(4):  # horizontal, vertical, diagonal, anti-diagonal
            if (state.corr[direction, player] == 4).any():
                return True
        return False

    def check_continuous_three(self, state):
        """
        Check for three continuous pieces of the current player.
        """
        player = state.current_player()
        for direction in range(4):  # horizontal, vertical, diagonal, anti-diagonal
            if (state.corr[direction, player] == 3).any():
                return True
        return False

    def uct_adp_search(self, state):
        # UCT-ADP search logic, incorporating pre_select_moves
        uct_adp = UCT_ADP(state)
        best_action = uct_adp.uct_search()
        
        return best_action


