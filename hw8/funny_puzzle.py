import heapq


def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader.

    INPUT:
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """

    manhattan_distance = 0
    for state_index, tile in enumerate(from_state):
        if tile != 0:
            x, y = divmod(state_index, 3)
            goal_index = to_state.index(tile)
            goal_x, goal_y = divmod(goal_index, 3)
            manhattan_distance += abs(x - goal_x) + abs(y - goal_y)
    return manhattan_distance


def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT:
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle.
    """

    successor_states = get_succ(state)
    for current_state in successor_states:
        print(current_state, "h={}".format(get_manhattan_distance(current_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT:
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below).
    """

    blank_indicies = [i for i, x in enumerate(state) if x == 0]
    successor_states = []
    seen_states = set()

    possible_moves = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4, 6],
        4: [1, 3, 5, 7],
        5: [2, 4, 8],
        6: [3, 7],
        7: [4, 6, 8],
        8: [5, 7],
    }
    for empty_idx in blank_indicies:
        for next_move in possible_moves[empty_idx]:
            if state[next_move] == 0:
                continue
            new_state = state[:]
            new_state[empty_idx], new_state[next_move] = (
                new_state[next_move],
                new_state[empty_idx],
            )
            new_state_tuple = tuple(new_state)
            if new_state_tuple not in seen_states:
                successor_states.append(new_state)
                seen_states.add(new_state_tuple)
    return sorted(successor_states)


def astar_search(priority_queue, cost_dict, state_details, max_length, goal_state):
    while priority_queue:
        current_f, current_state, current_g, parent_state = heapq.heappop(
            priority_queue
        )
        current_state_tuple = tuple(current_state)

        if (
            current_state_tuple in cost_dict
            and current_f > cost_dict[current_state_tuple]
        ):
            continue

        if current_state == goal_state:
            break

        for successor_state in get_succ(current_state):
            successor_state_tuple = tuple(successor_state)
            new_g = current_g + 1
            new_manhattan_distance = get_manhattan_distance(successor_state)
            new_cost = new_g + new_manhattan_distance

            if (
                successor_state_tuple not in cost_dict
                or new_cost < cost_dict[successor_state_tuple]
            ):
                heapq.heappush(
                    priority_queue,
                    (new_cost, successor_state, new_g, current_state_tuple),
                )
                cost_dict[successor_state_tuple] = new_cost
                state_details[successor_state_tuple] = (
                    new_g,
                    new_manhattan_distance,
                    current_state_tuple,
                )
        max_length = max(max_length, len(priority_queue))

    state_info_list = []
    current_state_tuple = tuple(goal_state)
    while current_state_tuple is not None:
        g, h, parent_state = state_details[current_state_tuple]
        state_info_list.append((list(current_state_tuple), h, g))
        current_state_tuple = parent_state

    return reversed(state_info_list), max_length


def solve(initial_state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT:
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """

    # This is a format helper.
    # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
    # define and compute max length
    # it can help to avoid any potential format issue.

    priority_queue = []
    initial_distance = get_manhattan_distance(initial_state)
    heapq.heappush(priority_queue, (initial_distance, initial_state, 0, None))

    cost_dict = {tuple(initial_state): initial_distance}
    state_details = {tuple(initial_state): (0, initial_distance, None)}

    state_info_list, max_length = astar_search(
        priority_queue=priority_queue,
        cost_dict=cost_dict,
        state_details=state_details,
        max_length=1,
        goal_state=goal_state,
    )

    # Print the solution path
    for state_info in state_info_list:
        current_state, h, move = state_info
        print(current_state, "h={}".format(h), "moves: {}".format(move))
    print("Max queue length: {}".format(max_length))


if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions.
    Note that this part will not be graded.
    """
    print_succ([2, 5, 1, 4, 0, 6, 7, 0, 3])
    print()

    print(
        get_manhattan_distance([2, 5, 1, 4, 0, 6, 7, 0, 3], [1, 2, 3, 4, 5, 6, 7, 0, 0])
    )
    print()

    solve([2, 5, 1, 4, 0, 6, 7, 0, 3])
    print()


### I used ChatGPT to get the psuedo code for A* search
### I also discussed with a friend to help me with the following
### Faced issue while generating the heuristic for leaf nodes
