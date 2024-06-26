import copy
import random


class TeekoPlayer:
    """An object representation for an AI game player for the game Teeko."""

    board = [[" " for j in range(5)] for i in range(5)]
    pieces = ["b", "r"]

    def __init__(self):
        """Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def check_drop_phase(self, state):
        piece_count = 0
        for row in state:
            for position in row:
                if position != " ":
                    piece_count += 1

        if piece_count == 8:
            return False
        else:
            return True

    def succ(self, state, piece):
        drop_phase = self.check_drop_phase(state)
        succ_list = []
        state_copy = copy.deepcopy(state)

        if drop_phase:
            for row in range(5):
                for col in range(5):
                    if state_copy[row][col] == " ":
                        state_copy[row][col] = piece
                        succ_list.append(state_copy)
                        state_copy = copy.deepcopy(state)
        else:
            for row in range(5):
                for col in range(5):
                    if state_copy[row][col] == piece:
                        if (
                            (row - 1 >= 0)
                            and (col - 1 >= 0)
                            and state_copy[row - 1][col - 1] == " "
                        ):
                            state_copy[row][col] = " "
                            state_copy[row - 1][col - 1] = piece
                            succ_list.append(state_copy)
                            state_copy = copy.deepcopy(state)

                        if (
                            (row - 1 >= 0)
                            and (col + 1 <= 4)
                            and state_copy[row - 1][col + 1] == " "
                        ):
                            state_copy[row][col] = " "
                            state_copy[row - 1][col + 1] = piece
                            succ_list.append(state_copy)
                            state_copy = copy.deepcopy(state)

                        if (
                            (row + 1 <= 4)
                            and (col - 1 >= 0)
                            and state_copy[row + 1][col - 1] == " "
                        ):
                            state_copy[row][col] = " "
                            state_copy[row + 1][col - 1] = piece
                            succ_list.append(state_copy)
                            state_copy = copy.deepcopy(state)

                        if (
                            (row + 1 <= 4)
                            and (col + 1 <= 4)
                            and state_copy[row + 1][col + 1] == " "
                        ):
                            state_copy[row][col] = " "
                            state_copy[row + 1][col + 1] = piece
                            succ_list.append(state_copy)
                            state_copy = copy.deepcopy(state)

                        if (row - 1 >= 0) and state_copy[row - 1][col] == " ":
                            state_copy[row][col] = " "
                            state_copy[row - 1][col] = piece
                            succ_list.append(state_copy)
                            state_copy = copy.deepcopy(state)

                        if (row + 1 <= 4) and state_copy[row + 1][col] == " ":
                            state_copy[row][col] = " "
                            state_copy[row + 1][col] = piece
                            succ_list.append(state_copy)
                            state_copy = copy.deepcopy(state)

                        if (col - 1 >= 0) and state_copy[row][col - 1] == " ":
                            state_copy[row][col] = " "
                            state_copy[row][col - 1] = piece
                            succ_list.append(state_copy)
                            state_copy = copy.deepcopy(state)

                        if (col + 1 <= 4) and state_copy[row][col + 1] == " ":
                            state_copy[row][col] = " "
                            state_copy[row][col + 1] = piece
                            succ_list.append(state_copy)
                            state_copy = copy.deepcopy(state)
        return succ_list

    def heuristic_game_value(self, state, piece):
        factor = 1 if piece == self.my_piece else -1
        if self.game_value(state) != 0:
            return self.game_value(state)

        max_row_score = 0
        for row in state:
            for i in range(2):
                row_score = 0
                if row[i] == piece:
                    row_score += 0.25
                elif row[i] != " ":
                    row_score -= 0.05

                if row[i + 1] == piece:
                    row_score += 0.25
                elif row[i + 1] != " ":
                    row_score -= 0.05

                if row[i + 2] == piece:
                    row_score += 0.25
                elif row[i + 2] != " ":
                    row_score -= 0.05

                if row[i + 3] == piece:
                    row_score += 0.25
                elif row[i + 3] != " ":
                    row_score -= 0.05

                if row_score > max_row_score:
                    max_row_score = row_score

        max_col_score = 0
        for col in range(5):
            for i in range(2):
                col_score = 0
                if state[i][col] == piece:
                    col_score += 0.25
                elif state[i][col] != " ":
                    col_score -= 0.05

                if state[i + 1][col] == piece:
                    col_score += 0.25
                elif state[i + 1][col] != " ":
                    col_score -= 0.05

                if state[i + 2][col] == piece:
                    col_score += 0.25
                elif state[i + 2][col] != " ":
                    col_score -= 0.05

                if state[i + 3][col] == piece:
                    col_score += 0.25
                elif state[i + 3][col] != " ":
                    col_score -= 0.05

                if col_score > max_col_score:
                    max_col_score = col_score

        max_diag1_score = 0
        for row in range(2):
            for col in range(2):
                diag1_score = 0
                if state[row][col] == piece:
                    diag1_score += 0.25
                elif state[row][col] != " ":
                    diag1_score -= 0.04

                if state[row + 1][col + 1] == piece:
                    diag1_score += 0.25
                elif state[row + 1][col + 1] != " ":
                    diag1_score -= 0.04

                if state[row + 2][col + 2] == piece:
                    diag1_score += 0.25
                elif state[row + 2][col + 2] != " ":
                    diag1_score -= 0.04

                if state[row + 3][col + 3] == piece:
                    diag1_score += 0.25
                elif state[row + 3][col + 3] != " ":
                    diag1_score -= 0.04

            if diag1_score > max_diag1_score:
                max_diag1_score = diag1_score

        max_diag2_score = 0
        for row in range(2):
            for col in range(3, 5):
                diag2_score = 0
                if state[row][col] == piece:
                    diag2_score += 0.25
                elif state[row][col] != " ":
                    diag2_score -= 0.04

                if state[row + 1][col - 1] == piece:
                    diag2_score += 0.25
                elif state[row + 1][col - 1] != " ":
                    diag2_score -= 0.04

                if state[row + 2][col - 2] == piece:
                    diag2_score += 0.25
                elif state[row + 2][col - 2] != " ":
                    diag2_score -= 0.04

                if state[row + 3][col - 3] == piece:
                    diag2_score += 0.25
                elif state[row + 3][col - 3] != " ":
                    diag2_score -= 0.04

            if diag2_score > max_diag2_score:
                max_diag2_score = diag2_score

        max_square_score = 0
        for row in range(3):
            for col in range(3):
                square_score = 0
                if state[row][col] == piece:
                    square_score += 0.25

                if state[row][col + 2] == piece:
                    square_score += 0.25

                if state[row + 2][col] == piece:
                    square_score += 0.25

                if state[row + 2][col + 2] == piece:
                    square_score += 0.25

                if state[row + 1][col + 1] != " ":
                    square_score -= 0.20

                if square_score > max_square_score:
                    max_square_score = square_score

        max_row_score += random.uniform(-0.02, 0.02)
        max_col_score += random.uniform(-0.02, 0.02)
        max_diag1_score += random.uniform(-0.02, 0.02)
        max_diag2_score += random.uniform(-0.02, 0.02)
        max_square_score += random.uniform(-0.02, 0.02)

        hval = max(
            max_row_score,
            max_col_score,
            max_diag1_score,
            max_diag2_score,
            max_square_score,
        )
        return factor * hval

    def max_value(self, alpha, beta, state, depth):
        if self.game_value(state) != 0:
            return self.game_value(state)

        if depth >= 3:
            return self.heuristic_game_value(state, self.my_piece)

        for successor in self.succ(state, self.my_piece):
            alpha = max(alpha, self.min_value(alpha, beta, successor, depth + 1))
            if alpha >= beta:
                return beta
        return alpha

    def min_value(self, alpha, beta, state, depth):
        if self.game_value(state) != 0:
            return self.game_value(state)

        if depth >= 3:
            return self.heuristic_game_value(state, self.opp)

        for successor in self.succ(state, self.opp):
            beta = min(beta, self.max_value(alpha, beta, successor, depth + 1))
            if alpha >= beta:
                return beta
        return beta

    def make_move(self, state):
        """Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        # drop_phase = True   # TODO: detect drop phase

        # if not drop_phase:
        #     # TODO: choose a piece to move and remove it from the board
        #     # (You may move this condition anywhere, just be sure to handle it)
        #     #
        #     # Until this part is implemented and the move list is updated
        #     # accordingly, the AI will not follow the rules after the drop phase!
        #     pass

        # # select an unoccupied space randomly
        # # TODO: implement a minimax algorithm to play better
        # move = []
        # (row, col) = (random.randint(0,4), random.randint(0,4))
        # while not state[row][col] == ' ':
        #     (row, col) = (random.randint(0,4), random.randint(0,4))

        # # ensure the destination (row,col) tuple is at the beginning of the move list
        # move.insert(0, (row, col))

        best_value = float("-inf")
        best_state = None
        for successor in self.succ(state, self.my_piece):
            current_value = self.min_value(float("-inf"), float("inf"), successor, 1)
            if current_value > best_value:
                best_value = current_value
                best_state = successor

        drop_phase = self.check_drop_phase(state)
        move = []
        if not drop_phase:
            for i in range(5):
                for j in range(5):
                    if state[i][j] != " " and state[i][j] != best_state[i][j]:
                        (source_row, source_col) = (i, j)
                        move.append((source_row, source_col))

        for i in range(5):
            for j in range(5):
                if state[i][j] == " " and state[i][j] != best_state[i][j]:
                    (row, col) = (i, j)
                    # ensure the destination (row,col) tuple is at the beginning of the move list
                    move.insert(0, (row, col))
        return move

    def opponent_move(self, move):
        """Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception("Illegal move: Can only move to an adjacent space")
        if self.board[move[0][0]][move[0][1]] != " ":
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = " "
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """Formatted printing for the board"""
        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != " " and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if (
                    state[i][col] != " "
                    and state[i][col]
                    == state[i + 1][col]
                    == state[i + 2][col]
                    == state[i + 3][col]
                ):
                    return 1 if state[i][col] == self.my_piece else -1

        # check \ diagonal wins
        for row in range(2):
            for col in range(2):
                if (
                    state[row][col] != " "
                    and state[row][col]
                    == state[row + 1][col + 1]
                    == state[row + 2][col + 2]
                    == state[row + 3][col + 3]
                ):
                    return 1 if state[row][col] == self.my_piece else -1

        # check / diagonal wins
        for row in range(2):
            for col in range(3, 5):
                if (
                    state[row][col] != " "
                    and state[row][col]
                    == state[row + 1][col - 1]
                    == state[row + 2][col - 2]
                    == state[row + 3][col - 3]
                ):
                    return 1 if state[row][col] == self.my_piece else -1

        # check 3x3 square corners wins
        for row in range(3):
            for col in range(3):
                if (
                    state[row][col] != " "
                    and state[row][col]
                    == state[row][col + 2]
                    == state[row + 2][col]
                    == state[row + 2][col + 2]
                ):
                    if state[row + 1][col + 1] == " ":
                        return 1 if state[row][col] == self.my_piece else -1

        return 0  # no winner yet


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print("Hello, this is Samaritan")
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:
        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(
                ai.my_piece
                + " moved at "
                + chr(move[0][1] + ord("A"))
                + str(move[0][0])
            )
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move(
                        [(int(player_move[1]), ord(player_move[0]) - ord("A"))]
                    )
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:
        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(
                ai.my_piece
                + " moved from "
                + chr(move[1][1] + ord("A"))
                + str(move[1][0])
            )
            print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move(
                        [
                            (int(move_to[1]), ord(move_to[0]) - ord("A")),
                            (int(move_from[1]), ord(move_from[0]) - ord("A")),
                        ]
                    )
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
