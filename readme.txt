Programming Language: Python
Version: Python 3.8.5


Code Structure:

The code is structured as follows:
- The main functionality is contained in a Python script named `expense_8_puzzle.py`.
- The Node class is defined at the beginning of the script.
- Various heuristic functions are defined for different search algorithms.
- Different search algorithms (BFS, UCS, DFS, IDS, DLS, A*, Greedy) are implemented as functions.
- Helper functions for puzzle actions and state manipulation are provided.


How to Run the Code:

1. Ensure you have Python 3.8.5 or a compatible version installed on your system.

2. Open a terminal and navigate to the directory containing the code.

3. Run the code using the following command from terminal:

python expense_8_puzzle.py start.txt goal.txt [search_algorithm] [trace]

Replace `[search_algorithm]` with one of the following options:
- bfs
- ucs
- dfs
- ids
- dls
- astar_h1
- astar_h2
- greedy_h1
- greedy_h2

Replace `[trace]` with a trace file name for trace generation.

4. The code will execute the selected search algorithm and provide the solution or indicate if no solution was found.


Note:
- Input puzzle states are provided in separate text files named `start.txt` and `goal.txt` in the same directory.


References:
- Referred geeksforgeeks.com to understand some functionalities.

ACS Omega Compatibility:
------------------------
This code is not specifically designed to run on ACS Omega. It is intended for local execution on a Python-compatible environment like VSCode.

For any questions or issues, please contact yxc5885@mavs.uta.edu
