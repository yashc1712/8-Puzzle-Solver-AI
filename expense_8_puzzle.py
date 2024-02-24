import sys
import heapq
from collections import deque

# Define the Node class
class Node:
    def __init__(self, state, parent=None, action=None, depth=0, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.depth = depth
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(tuple(map(tuple, self.state)))

    def __str__(self):
        return str(self.state)

# Define heuristic functions
def manhattan_distance(state, goal_state):
    distance = 0
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] != goal_state[i][j] and state[i][j] != 0:
                x, y = divmod(state[i][j] - 1, len(state))
                distance += abs(x - i) + abs(y - j)
    return distance

def heuristic_h1(node, goal_state):
    return manhattan_distance(node.state, goal_state)

def heuristic_h2(node, goal_state):
    # Modified heuristic: Misplaced tiles count
    misplaced = 0
    for i in range(len(node.state)):
        for j in range(len(node.state[i])):
            if node.state[i][j] != goal_state[i][j] and node.state[i][j] != 0:
                misplaced += 1
    return misplaced

# Define search algorithms

# Breadth-First Search (BFS)
def bfs(initial_node, goal_state, trace_file):
    frontier = deque([initial_node])
    explored = set()
    nodes_popped = 0
    nodes_expanded = 0
    max_fringe_size = 0
    trace_info = []  # Initialize trace_info as an empty list

    while frontier:
        node = frontier.popleft()
        nodes_popped += 1
        explored.add(tuple(map(tuple, node.state)))

        if node.state == goal_state:
            # Write trace information to the trace file
            if trace_file:
                with open(trace_file, "a") as trace:
                    trace.write("Command-Line Arguments: {}\n".format(sys.argv))
                    trace.write("Method Selected: BFS\n")
                    trace.write("Running BFS\n")
                    trace.write("Generating successors to < state = {}, action = {}, depth = {}, cost = {}, heuristic = {} >:\n".format(node.state, node.action, node.depth, node.cost, node.heuristic))
                    trace.write("\t{} successors generated\n".format(len(get_actions(node.state))))
                    trace.write("\tClosed: {}\n".format([str(s) for s in explored]))
                    trace.write("\tFringe: {}\n".format([str(n.state) for n in frontier]))
                    trace.write("\n")

            return node, nodes_popped, nodes_expanded, max_fringe_size, trace_info

        for action in get_actions(node.state):
            child_state = apply_action(node.state, action)
            if tuple(map(tuple, child_state)) not in explored:
                child_node = Node(child_state, node, action, node.depth + 1, node.cost + 1)
                frontier.append(child_node)
                nodes_expanded += 1
                max_fringe_size = max(max_fringe_size, len(frontier))

        # Append trace information for this loop to the trace_info list
        trace_info.append({
            "Fringe Contents": [str(n.state) for n in frontier],
            "Closed Set Contents": [str(s) for s in explored],
            "Nodes Expanded": nodes_expanded,
            "Nodes Popped": nodes_popped
        })

    return None, nodes_popped, nodes_expanded, max_fringe_size, trace_info

# Uniform-Cost Search (UCS)
def ucs(initial_node, goal_state, trace_file):
    frontier = [(initial_node.cost, initial_node)]  # Store the cost along with the node
    explored = set()
    nodes_popped = 0
    nodes_expanded = 0
    max_fringe_size = 0
    trace_info = []  # Initialize trace_info as an empty list

    while frontier:
        _, node = heapq.heappop(frontier)  # Extract the node with the lowest cost
        nodes_popped += 1
        explored.add(tuple(map(tuple, node.state)))

        if node.state == goal_state:
            # Write trace information to the trace file
            if trace_file:
                with open(trace_file, "a") as trace:
                    trace.write("Command-Line Arguments: {}\n".format(sys.argv))
                    trace.write("Method Selected: UCS\n")
                    trace.write("Running UCS\n")
                    trace.write("Generating successors to < state = {}, action = {}, depth = {}, cost = {}, heuristic = {} >:\n".format(node.state, node.action, node.depth, node.cost, node.heuristic))
                    trace.write("\t{} successors generated\n".format(len(get_actions(node.state))))
                    trace.write("\tClosed: {}\n".format([str(s) for s in explored]))
                    trace.write("\tFringe: {}\n".format([(cost, str(n.state)) for cost, n in frontier]))
                    trace.write("\n")

            return node, nodes_popped, nodes_expanded, max_fringe_size, trace_info

        for action in get_actions(node.state):
            child_state = apply_action(node.state, action)
            if tuple(map(tuple, child_state)) not in explored:
                child_node = Node(child_state, node, action, node.depth + 1, node.cost + 1)
                heapq.heappush(frontier, (child_node.cost, child_node))  # Store cost with the node
                nodes_expanded += 1
                max_fringe_size = max(max_fringe_size, len(frontier))

        # Append trace information for this loop to the trace_info list
        trace_info.append({
            "Fringe Contents": [(cost, str(n.state)) for cost, n in frontier],
            "Closed Set Contents": [str(s) for s in explored],
            "Nodes Expanded": nodes_expanded,
            "Nodes Popped": nodes_popped
        })

    return None, nodes_popped, nodes_expanded, max_fringe_size, trace_info


# Depth-First Search (DFS)
def dfs(initial_node, goal_state, trace_file):
    stack = [(initial_node, 0)]  # Stack stores both the node and its depth
    explored = set()
    nodes_popped = 0
    nodes_expanded = 0
    max_fringe_size = 0
    trace_info = []

    while stack:
        node, current_depth = stack.pop()
        nodes_popped += 1
        explored.add(tuple(map(tuple, node.state)))

        if node.state == goal_state:
            
            return node, nodes_popped, nodes_expanded, max_fringe_size, trace_info

        if current_depth < len(initial_node.state) * len(initial_node.state[0]):
            for action in reversed(get_actions(node.state)):
                child_state = apply_action(node.state, action)
                if tuple(map(tuple, child_state)) not in explored:
                    child_node = Node(child_state, node, action, node.depth + 1, node.cost + 1)
                    stack.append((child_node, current_depth + 1))
                    nodes_expanded += 1
                    max_fringe_size = max(max_fringe_size, len(stack))

            trace_info.append({
                "Fringe Contents": [str(n[0].state) for n in stack],
                "Closed Set Contents": [str(s) for s in explored],
                "Nodes Expanded": nodes_expanded,
                "Nodes Popped": nodes_popped
            })

    return None, nodes_popped, nodes_expanded, max_fringe_size, trace_info


# Depth-Limited Search (DLS)
def dls(initial_node, goal_state, depth_limit, trace_file):
    stack = [(initial_node, 0)]  # Store the node along with its current depth
    explored = set()
    nodes_popped = 0
    nodes_expanded = 0
    max_fringe_size = 0
    trace_info = []  # Initialize trace_info as an empty list

    while stack:
        node, current_depth = stack.pop()  # Extract the node and its current depth
        nodes_popped += 1
        explored.add(tuple(map(tuple, node.state)))

        if node.state == goal_state:
            # Write trace information to the trace file
            if trace_file:
                with open(trace_file, "a") as trace:
                    trace.write("Command-Line Arguments: {}\n".format(sys.argv))
                    trace.write("Method Selected: DLS\n")
                    trace.write("Running DLS with depth limit {}\n".format(depth_limit))
                    trace.write("Generating successors to < state = {}, action = {}, depth = {}, cost = {}, heuristic = {} >:\n".format(node.state, node.action, node.depth, node.cost, node.heuristic))
                    trace.write("\t{} successors generated\n".format(len(get_actions(node.state))))
                    trace.write("\tClosed: {}\n".format([str(s) for s in explored]))
                    trace.write("\tFringe: {}\n".format([(str(n[0].state), n[1]) for n in stack]))
                    trace.write("\n")

            return node, nodes_popped, nodes_expanded, max_fringe_size, trace_info

        if current_depth < depth_limit:
            for action in reversed(get_actions(node.state)):
                child_state = apply_action(node.state, action)
                if tuple(map(tuple, child_state)) not in explored:
                    child_node = Node(child_state, node, action, node.depth + 1, node.cost + 1)
                    stack.append((child_node, current_depth + 1))  # Store the node and its depth
                    nodes_expanded += 1
                    max_fringe_size = max(max_fringe_size, len(stack))

        # Append trace information for this loop to the trace_info list
        trace_info.append({
            "Fringe Contents": [(str(n[0].state), n[1]) for n in stack],
            "Closed Set Contents": [str(s) for s in explored],
            "Nodes Expanded": nodes_expanded,
            "Nodes Popped": nodes_popped
        })

    return None, nodes_popped, nodes_expanded, max_fringe_size, trace_info

# Iterative Deepening Search (IDS)
def ids(initial_node, goal_state, trace_file):
    nodes_popped = 0
    nodes_expanded = 0
    max_fringe_size = 0
    trace_info = []  # Initialize trace_info as an empty list

    for depth_limit in range(sys.maxsize):
        result, popped, expanded, fringe, trace = dls(initial_node, goal_state, depth_limit, trace_file)
        nodes_popped += popped
        nodes_expanded += expanded
        max_fringe_size = max(max_fringe_size, fringe)

        # Write trace information to the trace file
        if trace_file:
            with open(trace_file, "a") as trace:
                trace.write("Command-Line Arguments: {}\n".format(sys.argv))
                trace.write("Method Selected: IDS\n")
                trace.write("Running IDS with depth limit {}\n".format(depth_limit))
                trace.write("Nodes Expanded: {}\n".format(expanded))
                trace.write("Nodes Popped: {}\n".format(popped))
                trace.write("\n")

        if result:
            return result, nodes_popped, nodes_expanded, max_fringe_size, trace_info

    return None, nodes_popped, nodes_expanded, max_fringe_size, trace_info

# A* Search
def a_star(initial_node, goal_state, heuristic, trace_file):
    frontier = [initial_node]
    explored = set()
    nodes_popped = 0
    nodes_expanded = 0
    max_fringe_size = 0
    trace_info = []  # Initialize trace_info as an empty list

    while frontier:
        node = heapq.heappop(frontier)
        nodes_popped += 1
        explored.add(tuple(map(tuple, node.state)))

        if node.state == goal_state:
            # Write trace information to the trace file
            if trace_file:
                with open(trace_file, "a") as trace:
                    trace.write("Command-Line Arguments: {}\n".format(sys.argv))
                    trace.write("Method Selected: A*\n")
                    trace.write("Running A* using {}\n".format(heuristic.__name__))
                    trace.write("Generating successors to < state = {}, action = {}, depth = {}, cost = {}, heuristic = {} >:\n".format(node.state, node.action, node.depth, node.cost, node.heuristic))
                    trace.write("\t{} successors generated\n".format(len(get_actions(node.state))))
                    trace.write("\tClosed: {}\n".format([str(s) for s in explored]))
                    trace.write("\tFringe: {}\n".format([str(n.state) for n in frontier]))
                    trace.write("\n")

            return node, nodes_popped, nodes_expanded, max_fringe_size, trace_info

        for action in get_actions(node.state):
            child_state = apply_action(node.state, action)
            if tuple(map(tuple, child_state)) not in explored:
                child_node = Node(child_state, node, action, node.depth + 1, node.cost + 1)
                child_node.heuristic = heuristic(child_node, goal_state)
                heapq.heappush(frontier, child_node)
                nodes_expanded += 1
                max_fringe_size = max(max_fringe_size, len(frontier))

        # Append trace information for this loop to the trace_info list
        trace_info.append({
            "Fringe Contents": [str(n.state) for n in frontier],
            "Closed Set Contents": [str(s) for s in explored],
            "Nodes Expanded": nodes_expanded,
            "Nodes Popped": nodes_popped
        })

    return None, nodes_popped, nodes_expanded, max_fringe_size, trace_info

# Greedy Search
def greedy(initial_node, goal_state, heuristic, trace_file):
    frontier = [(heuristic(initial_node, goal_state), initial_node)]
    explored = set()
    nodes_popped = 0
    nodes_expanded = 0
    max_fringe_size = 0
    trace_info = []  # Initialize trace_info as an empty list

    while frontier:
        _, node = heapq.heappop(frontier)
        nodes_popped += 1
        explored.add(tuple(map(tuple, node.state)))

        if node.state == goal_state:
            # Write trace information to the trace file
            if trace_file:
                with open(trace_file, "a") as trace:
                    trace.write("Command-Line Arguments: {}\n".format(sys.argv))
                    trace.write("Method Selected: Greedy\n")
                    trace.write("Running Greedy using {}\n".format(heuristic.__name__))
                    trace.write("Generating successors to < state = {}, action = {}, depth = {}, cost = {}, heuristic = {} >:\n".format(node.state, node.action, node.depth, node.cost, node.heuristic))
                    trace.write("\t{} successors generated\n".format(len(get_actions(node.state))))
                    trace.write("\tClosed: {}\n".format([str(s) for s in explored]))
                    trace.write("\tFringe: {}\n".format([(heuristic(n[1], goal_state), str(n[1].state)) for n in frontier]))
                    trace.write("\n")

            return node, nodes_popped, nodes_expanded, max_fringe_size, trace_info

        for action in get_actions(node.state):
            child_state = apply_action(node.state, action)
            if tuple(map(tuple, child_state)) not in explored:
                child_node = Node(child_state, node, action, node.depth + 1, node.cost + 1)
                child_node.heuristic = heuristic(child_node, goal_state)

                heapq.heappush(frontier, (heuristic(child_node, goal_state), child_node))
                nodes_expanded += 1
                max_fringe_size = max(max_fringe_size, len(frontier))

        # Append trace information for this loop to the trace_info list
        trace_info.append({
            "Fringe Contents": [(heuristic(n[1], goal_state), str(n[1].state)) for n in frontier],
            "Closed Set Contents": [str(s) for s in explored],
            "Nodes Expanded": nodes_expanded,
            "Nodes Popped": nodes_popped
        })

    return None, nodes_popped, nodes_expanded, max_fringe_size, trace_info

# Helper functions
def get_actions(state):
    actions = []
    empty_i, empty_j = find_empty_tile(state)
    if empty_i > 0:
        actions.append("Up")
    if empty_i < 2:
        actions.append("Down")
    if empty_j > 0:
        actions.append("Left")
    if empty_j < 2:
        actions.append("Right")
    return actions

def apply_action(state, action):
    empty_i, empty_j = find_empty_tile(state)
    new_state = [list(row) for row in state]

    if action == "Up":
        new_state[empty_i][empty_j], new_state[empty_i - 1][empty_j] = new_state[empty_i - 1][empty_j], 0
    elif action == "Down":
        new_state[empty_i][empty_j], new_state[empty_i + 1][empty_j] = new_state[empty_i + 1][empty_j], 0
    elif action == "Left":
        new_state[empty_i][empty_j], new_state[empty_i][empty_j - 1] = new_state[empty_i][empty_j - 1], 0
    elif action == "Right":
        new_state[empty_i][empty_j], new_state[empty_i][empty_j + 1] = new_state[empty_i][empty_j + 1], 0

    return new_state


def find_empty_tile(state):
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] == 0:
                return i, j

# Main function
def main():
    if len(sys.argv) != 5:
        print("Usage: expense_8_puzzle.py start.txt goal.txt <search_algorithm> <trace>")
        sys.exit(1)

    start_file = sys.argv[1]
    goal_file = sys.argv[2]
    search_algorithm = sys.argv[3]
    trace_option = sys.argv[4]

    # Read start and goal states from files
    start_state = read_state(start_file)
    goal_state = read_state(goal_file)

    # Create initial node
    initial_node = Node(start_state)

    # Select the search algorithm and run it
    if search_algorithm == "bfs":
        solution, nodes_popped, nodes_expanded, max_fringe_size, trace_info = bfs(initial_node, goal_state, trace_option)
    elif search_algorithm == "ucs":
        solution, nodes_popped, nodes_expanded, max_fringe_size, trace_info = ucs(initial_node, goal_state, trace_option)
    elif search_algorithm == "dfs":
        solution, nodes_popped, nodes_expanded, max_fringe_size, trace_info = dfs(initial_node, goal_state, trace_option)
    elif search_algorithm == "ids":
        solution, nodes_popped, nodes_expanded, max_fringe_size, trace_info = ids(initial_node, goal_state, trace_option)
    elif search_algorithm == "dls":
        try:
            depth_limit = int(input("Enter the depth limit: "))
        except ValueError:
            print("Invalid depth limit for DLS.")
            sys.exit(1)
        solution, nodes_popped, nodes_expanded, max_fringe_size, trace_info = dls(initial_node, goal_state, depth_limit, trace_option)
    elif search_algorithm == "astar_h1":
        solution, nodes_popped, nodes_expanded, max_fringe_size, trace_info = a_star(initial_node, goal_state, heuristic_h1, trace_option)
    elif search_algorithm == "astar_h2":
        solution, nodes_popped, nodes_expanded, max_fringe_size, trace_info = a_star(initial_node, goal_state, heuristic_h2, trace_option)
    elif search_algorithm == "greedy_h1":
        solution, nodes_popped, nodes_expanded, max_fringe_size, trace_info = greedy(initial_node, goal_state, heuristic_h1, trace_option)
    elif search_algorithm == "greedy_h2":
        solution, nodes_popped, nodes_expanded, max_fringe_size, trace_info = greedy(initial_node, goal_state, heuristic_h2, trace_option)
    else:
        print("Invalid search algorithm. Please choose from 'bfs', 'ucs', 'dfs', 'ids', 'astar_h1', 'astar_h2', 'greedy_h1', 'greedy_h2'")
        sys.exit(1)

    # Print the solution or write it to a file
    if solution:
        print("Nodes Popped: {}".format(nodes_popped))
        print("Nodes Expanded: {}".format(nodes_expanded))
        print("Nodes Generated: {}".format(nodes_popped + nodes_expanded))
        print("Max Fringe Size: {}".format(max_fringe_size))
        print("Solution Found at depth {} with cost of {}.".format(solution.depth, solution.cost))
        print("Steps:")

        actions = []
        while solution:
            if solution.action:
                actions.insert(0, (solution.action, solution.state))  # Store both action and state
            solution = solution.parent

        current_state = start_state
        for action, state in actions:
            # Get the tile number from the state
            empty_i, empty_j = find_empty_tile(current_state)
            if action == "Up":
                new_i, new_j = empty_i - 1, empty_j
            elif action == "Down":
                new_i, new_j = empty_i + 1, empty_j
            elif action == "Left":
                new_i, new_j = empty_i, empty_j - 1
            elif action == "Right":
                new_i, new_j = empty_i, empty_j + 1
            tile_number = current_state[new_i][new_j]

            # Print the move
            print("Move {} {}".format(tile_number, action))
            # Update the current state
            current_state = state

    else:
        print("No solution found.")

        
def read_state(file_name):
    with open(file_name, "r") as file:
        state = []
        for line in file:
            line = line.strip()
            if line == "END OF FILE":
                break
            row = [int(x) for x in line.split()]
            state.append(row)
    return state


if __name__ == "__main__":
    main()