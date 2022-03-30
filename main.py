import numpy as np
import random
import pygame
import sys
import math


# Colors for the game
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0,255,0)
YELLOW = (255, 255, 0)

# Size of the board
ROW_COUNT = 6
COLUMN_COUNT = 7

# Player numbers
PLAYER = 0
AI = 1

# Piece symbols
EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

# More details
WINDOW_LENGTH = 4
INF = 10000
WIN_SCORE = 100
LOSE_SCORE = -100

# Pygame design parameters
SQUARESIZE = 100
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE
size = (width, height)
RADIUS = int(SQUARESIZE / 2 - 5)



###################################################
               # Util functions #
###################################################

def create_board():
    """
    The function initialized the board in the start of the game
    :return: matrix of zeros in the size of the game board
    """
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board


def get_next_open_row(board, col):
    """
    The function find the first row that open in the column
    :param board: the current state of the game
    :param col: the requested column
    :return: the first row that open in requested column
    """
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def drop_piece(board, row, col, piece):
    """
    The function put the piece in request index
    :param board: the current state of the game
    :param row: the requested row
    :param col: the requested column
    :param piece: int number that represent type of piece (1 for player, 2 for AI)
    :return: Nothing, it update the board after the drop
    """
    board[row][col] = piece


def is_valid_location(board, col):
    """
    The function check if the player could throw piece to the requested column
    :param board: the current state of the game
    :param col: the requested column
    :return: a boolean value (true\false) if the player could throw the piece
    """
    return board[ROW_COUNT - 1][col] == 0


def print_board(board):
    """
    The function print the board
    :param board: the current state of the game
    :return: print
    """
    print(np.flip(board, 0))


def winning_move(board, piece):
    """
    The function check if there is a vec in board that have 4 same input piece (a vec to win)
    :param board: the current state of the game
    :param piece: int number that represent type of piece (1 for player, 2 for AI)
    :return: a boolean value (true\false) if there is a win for the input piece
    """
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                c + 3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True

    return False


def get_valid_locations(board):
    """
    The function get all the columns that not full so the player could drop to them piece
    :param board: the current state of the game
    :return: list of columns number that no fulled yet
    """
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def is_terminal_node(board):
    """
    the function check if there a win at all
    :param board: the current state of the game
    :return: a boolean value (true\false) if there is a win
    """
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE)


def draw_board(board):
    """
    The function draw the board in pygame graphic interface
    :param board: the current state of the game
    :return: the graphic interface of the game
    """
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (
            int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (
                int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, YELLOW, (
                int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()



###################################################
# Class of the game, Heuristic function, #
# and Minimax algo' with alpha beta pruning #
###################################################

class ConnectFour:

    """
    This class represent the state space of the game.
    the class get the current board, and the current turn (0 for player, 1 to AI)
    the class save this parameters and the next methods:
    actions: return the columns that not fulled (legall actions to play)
    succ: get a col and return a object from this class with the new board that create and the next turn (the opp turn)
    is_goal: return a boolean value (true\false) if is a win board
    print_state: return print of the board
    """

    def __init__(self, state, turn):
        self.state = state.copy()
        self.turn = turn

    def actions(self):
        return get_valid_locations(self.state)

    def succ(self, col):
        piece = AI_PIECE
        if self.turn == PLAYER:
            piece = PLAYER_PIECE

        row = get_next_open_row(self.state, col)
        s_copy = self.state.copy()
        drop_piece(s_copy, row, col, piece)

        opp_turn = abs(self.turn - 1)
        succ = ConnectFour(s_copy, opp_turn)
        return succ

    def is_goal(self):
        return is_terminal_node(self.state)

    def print_state(self):
        return print_board(self.state)


def evaluate_window(window, piece):
    """
    The function get a vec in length 4 and piece and evalute its score by the method we define
    :param window: a vec in length 4 from the board
    :param piece: int number that represent type of piece (1 for player, 2 for AI)
    :return: a score of the evaluation of the window according to the player piece
    """
    score = 0

    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    # Check if this window can lead to win
    if np.count_nonzero(window - (opp_piece * np.ones(4))) == 4:
        score = np.count_nonzero(window)    # The number of same player piece in the window

    return score


def Huristic_Function(state):
    """
    The function get a state of the board and return a heuristic value to the state that represent how much the AI
    close to win the game (high value mean the AI very close to win)
    the function calculate this value by evaluate all the windows for AI and reduce the evaluation of all the windows
    for the player
    :param state: the current state of the game
    :return: the heuristic function of the state
    """
    score = 0

    ## Score Horizontal
    for r in range(ROW_COUNT):
        row_array = state[r, :]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, AI_PIECE) - evaluate_window(window, PLAYER_PIECE)

    ## Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = state[:, c]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, AI_PIECE) - evaluate_window(window, PLAYER_PIECE)

    ## Score posiive sloped diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = np.array([state[r + i][c + i] for i in range(WINDOW_LENGTH)])
            score += evaluate_window(window, AI_PIECE) - evaluate_window(window, PLAYER_PIECE)

    ## Score negative sloped diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = np.array([state[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)])
            score += evaluate_window(window, AI_PIECE) - evaluate_window(window, PLAYER_PIECE)

    return score


def minimax(board, depth, alpha, beta):
    """
    The function apply the minimax algorithm with alpha beta pruning to chosen depth
    :param board: the current state of the game
    :param depth: The number of rounds to check forward
    :param alpha: the alpha value from alpha beta pruning
    :param beta: the beta value from alpha beta pruning
    :return: the best column to act and the score of the minimax we get
    """
    valid_locations = board.actions()

    if board.is_goal():
        if board.turn == PLAYER:  # This is opp turn so it's mean that AI win
            return (None, WIN_SCORE - 1/(depth+1))
        elif board.turn == AI:    # This is opp turn so it's mean that PlAYER win
            return (None, LOSE_SCORE + 1/(depth+1))

    if len(valid_locations) == 0:  # Game is over, no more valid moves, is a tie
        return (None, 0)

    if depth == 0:  # Depth is zero
        return (None, Huristic_Function(board.state))

    if board.turn:  # AI turn (the maximizer)
        value = -INF
        column = random.choice(valid_locations)
        for col in valid_locations:
            new_score = minimax(board.succ(col), depth, alpha, beta)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else:  # player turn (the Minimizer)
        value = INF
        column = random.choice(valid_locations)
        for col in valid_locations:
            new_score = minimax(board.succ(col), depth - 1, alpha, beta)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value



###################################################
               # main function #
###################################################

if __name__ == "__main__":

    DEPTH = 2
    round = 0
    start_board = create_board()
    game_over = False

    pygame.init()
    screen = pygame.display.set_mode(size)
    draw_board(start_board)
    pygame.display.update()
    myfont = pygame.font.SysFont("monospace", 75)

    turn = random.randint(PLAYER, AI)
    while not game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                posx = event.pos[0]

                if turn == PLAYER:
                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)

            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))

                # Ask for Player 1 Input
                if turn == PLAYER:
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))

                    if is_valid_location(start_board, col):
                        row = get_next_open_row(start_board, col)
                        drop_piece(start_board, row, col, PLAYER_PIECE)

                        if winning_move(start_board, PLAYER_PIECE):
                            label = myfont.render("Player 1 wins!!", 1, RED)
                            screen.blit(label, (40, 10))
                            game_over = True

                        turn = AI
                        draw_board(start_board)

                        if len(get_valid_locations(start_board)) == 0:
                            label = myfont.render("Is a tie", 1, GREEN)
                            screen.blit(label, (40, 10))
                            game_over = True


        # AI turn
        if turn == AI and not game_over:
            round += 1  # Count the round
            # Check our progress position in the game, for initialize more deep checks
            if round >= 10:
                DEPTH = 4
            elif round >= 4:
                DEPTH = 3

            # DO minimax
            col, minimax_score = minimax(ConnectFour(start_board, AI), DEPTH, -INF, INF)
            if is_valid_location(start_board, col):

                row = get_next_open_row(start_board, col)
                drop_piece(start_board, row, col, AI_PIECE)

                if winning_move(start_board, AI_PIECE):
                    label = myfont.render("Player 2 wins!!", 1, YELLOW)
                    screen.blit(label, (40, 10))
                    game_over = True

                turn = PLAYER
                draw_board(start_board)

        if len(get_valid_locations(start_board)) == 0:
            label = myfont.render("Is a tie", 1, GREEN)
            screen.blit(label, (40, 10))
            game_over = True

        if game_over:
            pygame.time.wait(3000)


