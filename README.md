# CPEN 513 Assignment 3
CPEN 513 CAD Algorithms for Integrated Circuits Assignment 3: Branch-and-Bound based Bi-partitioning

# Usage
Run src/main.py with a Python 3.9 interpreter

A partitioning window should be displayed upon running the file. This may take a few seconds,
as the program performs an initial partition by default. 

Press '0' to run the entire algorithm to completion. The GUI may freeze during this,
but information should be continually logged to the console.

# GUI + Console Explanation
The initial partition will be displayed in the window at program start.
Each node will be placed in a white block with the ID of the node displayed. The ID text is black by default, 
but will be blue if the node is a source for a net that crosses the partition. A red line
is drawn for each source-sink connection. Each of the two columns of white blocks 
represents half of the bi-partition.
The black blocks are dead space included to make the solution easier to read.

The final solution will be displayed in the GUI at program completion. The console will log initial and final solution costs, 
runtime info, the final solution partition, and whether the final solution is an optimal solution or a heuristic solution.
