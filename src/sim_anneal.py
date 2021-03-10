"""
Solution to UBC CPEN 513 Assignment 3.
Implements branch-and-bound based bi-partitioning.
Uses Tkinter for GUI.
"""

import os
import random
import time
import matplotlib.pyplot as plt
from tkinter import *
from math import exp, sqrt, ceil

# Constants
file_name = "test.txt"
FILE_DIR = "../benchmarks/"

# Hyperparameters
cooling_factor = 0.8  # Coefficient for rate of anneal cooling
initial_temp_factor = 10  # Coefficient for anneal initial temperature
moves_per_temp_factor = 75  # Coefficient for number of moves to be performed at each temperature
COST_TRANSITION_RATIO = 0.8  # Ratio for determining when to start using a range window for moves
TEMP_EXIT_RATIO = 0.002  # Ratio for determining exit condition based on temperature
COST_EXIT_RATIO = 0.005  # Ratio for determining exit condition based on cost
MOVE_SAMPLE_SIZE = 50  # Initial number of moves to be performed to determine cost variance of moves
hyperparam_string = str(cooling_factor) + "-" + str(initial_temp_factor) + "-" + str(moves_per_temp_factor) + "-"

# Global variables
num_cells_to_place = 0  # Number of cells in the circuit to be placed
num_cell_connections = 0  # Number of connections to be routed, summed across all cells/nets
grid_width = 2  # Width of the placement grid
grid_height = 0  # Height of the placement grid
half_grid_max_dim = 0  # Larger of width/height
cell_dict = {}  # Dictionary of all cells, key is cell ID
net_dict = {}  # Dictionary of all nets, key is net ID
partition_grid = []  # 2D list of sites for placement
partitioning_done = False  # Is the placement complete?
cost_history = []  # History of costs at each temperature
iter_history = []  # History of cumulative iterations performed at each temperature
temperature_history = []  # History of exact temperature values
acceptance_history = []  # History of number of accepted moves
root = None  # Tkinter root
unique_line_list = []  # List of unique lines across multiple nets
# Simulated Annealing
sa_temp = -1  # SA temperature
sa_initial_temp = -1  # Starting SA temperature
iters_per_temp = -1  # Number of iterations to perform at each temperature
iters_this_temp = 0  # Number of iterations performed at the current temperature
initial_cost = -1  # Cost of initial netlist placement
current_cost = 0  # The estimated cost of the current placement
prev_temp_cost = -1  # Cost at the end of exploring the previous temperature
prev_temp_cost_ratio = float("inf")  # Cost ratio at the end of exploring the previous temperature
total_iters = 0  # Cumulative number of iterations performed throughout program run
acceptances_this_temp = 0  # Number of accepted moves at the current temperature
range_window_half_length = -1  # Half the length of a side of the (square) range window


class Site:
    """
    A placement site/slot for a cell to inhabit
    """
    def __init__(self, x: int, y: int):
        self.x = x  # x location
        self.y = y  # y location
        self.canvas_id = -1  # ID of corresponding rectangle in Tkinter canvas
        self.canvas_centre = (-1, -1)  # Geometric centre of Tkinter rectangle
        self.isOccupied = False  # Is the site occupied by a cell?
        self.occupant = None  # Reference to occupant cell
        pass


class Cell:
    """
    A single cell
    """
    def __init__(self, cell_id):
        self.id = cell_id  # Identifier
        self.isPlaced = False  # Has this cell been placed into a site?
        self.site = None  # Reference to the site this cell occupies
        self.nets = []  # Nets this cell is a part of
        pass


class Line:
    """
    A wrapper class for Tkinter lines
    """
    def __init__(self, source: Cell, sink: Cell, canvas_id: int):
        self.source = source  # Reference to source cell
        self.sink = sink  # Reference to sink cell
        self.canvas_id = canvas_id  # Tkinter ID of line


class Net:
    """
    A collection of cells to be connected during routing
    """
    def __init__(self, net_id: int, num_cells: int):
        self.id = net_id  # Identifier
        self.num_cells = num_cells  # Number of cells in this net
        self.source = None  # Reference to source cell
        self.sinks = []  # References to sink cells
        self.lines = []  # References to Lines in this net
        pass


def reset_globals():
    """
    Reset (most) global variables.
    Used for repeated calls to this script from a parent script.
    """
    global num_cells_to_place
    global num_cell_connections
    global grid_width
    global grid_height
    global cell_dict
    global net_dict
    global partition_grid
    global partitioning_done
    global cost_history
    global iter_history
    global temperature_history
    global root
    global unique_line_list
    global sa_temp
    global sa_initial_temp
    global iters_per_temp
    global iters_this_temp
    global current_cost
    global prev_temp_cost
    global total_iters

    num_cells_to_place = 0
    num_cell_connections = 0
    grid_width = 0
    grid_height = 0
    cell_dict = {}
    net_dict = {}
    partition_grid = []
    partitioning_done = False
    cost_history = []
    iter_history = []
    temperature_history = []
    root = None
    unique_line_list = []
    sa_temp = -1
    sa_initial_temp = -1
    iters_per_temp = -1
    iters_this_temp = -1
    current_cost = 0
    prev_temp_cost = 0
    total_iters = 0


def quick_partition(f_name, cool_fact, init_temp_fact, move_p_t_fact):
    """
    Perform a partition without a GUI. Automatically exits after saving data.
    For experimentation.
    """
    global FILE_DIR
    global file_name
    global num_cells_to_place
    global num_cell_connections
    global grid_width
    global grid_height
    global partition_grid
    global cooling_factor
    global initial_temp_factor
    global moves_per_temp_factor
    global hyperparam_string

    reset_globals()

    random.seed(0)  # Set random seed

    file_name = f_name
    cooling_factor = cool_fact
    initial_temp_factor = init_temp_fact
    moves_per_temp_factor = move_p_t_fact
    hyperparam_string = str(cooling_factor) + "-" + str(initial_temp_factor) + "-" + str(moves_per_temp_factor) + "-"

    print("Running: " + file_name + "-" + str(cooling_factor) + "-" + str(initial_temp_factor) + "-" +
          str(moves_per_temp_factor))

    file_path = FILE_DIR + f_name
    script_path = os.path.dirname(__file__)
    true_path = os.path.join(script_path, file_path)
    routing_file = open(true_path, "r")

    # Setup the routing grid/array
    partition_grid = create_partition_grid(routing_file)

    # Perform initial placement
    initial_partition(None)

    partition_to_completion(None)

    
def partition(f_name: str):
    """
    Perform anneal with a GUI.
    :param f_name: Name of file to open
    :return: void
    """
    global FILE_DIR
    global file_name
    global num_cells_to_place
    global num_cell_connections
    global grid_width
    global grid_height
    global partition_grid
    global root

    random.seed(0)  # Set random seed

    # Determine file to open
    file_name = f_name
    file_path = FILE_DIR + file_name
    script_path = os.path.dirname(__file__)
    true_path = os.path.join(script_path, file_path)
    routing_file = open(true_path, "r")

    # Setup the routing grid/array
    partition_grid = create_partition_grid(routing_file)

    # Create routing canvas in Tkinter
    root = Tk()
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    site_length = 20
    partition_canvas = Canvas(root, bg='white', width=grid_width*site_length, height=grid_height*site_length)
    partition_canvas.grid(column=0, row=0, sticky=(N, W, E, S))
    for y in range(grid_height):
        x = 0
        # Add a cell site rectangle to the canvas
        top_left_x = site_length * x
        top_left_y = site_length * y
        bottom_right_x = top_left_x + site_length
        bottom_right_y = top_left_y + site_length
        rectangle_colour = "white"
        rectangle_coords = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        site = partition_grid[y][x]
        site.canvas_id = partition_canvas.create_rectangle(rectangle_coords, fill=rectangle_colour)
        site.canvas_centre = ((top_left_x+bottom_right_x)/2, (top_left_y+bottom_right_y)/2)
        for x in range(1, grid_height-1):
            # Add a cell site rectangle to the canvas
            top_left_x = site_length * x
            top_left_y = site_length * y
            bottom_right_x = top_left_x + site_length
            bottom_right_y = top_left_y + site_length
            rectangle_colour = "black"
            rectangle_coords = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
            partition_canvas.create_rectangle(rectangle_coords, fill=rectangle_colour)

        x = grid_height-1
        # Add a cell site rectangle to the canvas
        top_left_x = site_length * x
        top_left_y = site_length * y
        bottom_right_x = top_left_x + site_length
        bottom_right_y = top_left_y + site_length
        rectangle_colour = "white"
        rectangle_coords = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        site = partition_grid[y][1]
        site.canvas_id = partition_canvas.create_rectangle(rectangle_coords, fill=rectangle_colour)
        site.canvas_centre = ((top_left_x + bottom_right_x) / 2, (top_left_y + bottom_right_y) / 2)

    # Perform initial placement
    initial_partition(partition_canvas)

    # Event bindings and Tkinter start
    partition_canvas.focus_set()
    partition_canvas.bind("<Key>", lambda event: key_handler(partition_canvas, event))
    root.mainloop()


def initial_partition(partition_canvas):
    """
    Perform an initial placement prior to Simulated Annealing
    :param partition_canvas: Tkinter canvas
    """
    global partition_grid
    global current_cost
    global sa_temp
    global moves_per_temp_factor
    global num_cells_to_place
    global iters_per_temp
    global sa_initial_temp
    global cost_history
    global iter_history
    global initial_cost

    # Check if there are enough sites for the requisite number of cells
    if num_cells_to_place > (grid_width*grid_height):
        print("ERROR: Not enough space to place this circuit!")
        exit()

    # Get a list of all sites to place cells into
    free_sites = []
    for x in range(grid_width):
        for y in range(grid_height):
            free_sites.append((x, y))
    random.shuffle(free_sites)  # Randomize order to avoid undesired initial placement structure

    for net in net_dict.values():
        # Place the net's source
        if not net.source.isPlaced:
            place_x, place_y = free_sites.pop()
            placement_site = partition_grid[place_y][place_x]
            placement_site.occupant = net.source
            net.source.site = placement_site
            placement_site.isOccupied = True
            net.source.isPlaced = True

        # Place the net's sinks
        for sink in net.sinks:
            if not sink.isPlaced:
                place_x, place_y = free_sites.pop()
                placement_site = partition_grid[place_y][place_x]
                placement_site.occupant = sink
                sink.site = placement_site
                placement_site.isOccupied = True
                sink.isPlaced = True

        # Draw net on canvas
        if partition_canvas is not None:
            draw_net(partition_canvas, net)


def draw_net(partition_canvas, net):
    """
    Draw a net on the canvas from scratch
    """
    global unique_line_list

    # Check that net is fully placed
    if not net.source.isPlaced:
        return
    for sink in net.sinks:
        if not sink.isPlaced:
            return

    # Draw line between cells
    for sink in net.sinks:
        if partition_canvas is not None:
            line_id = draw_line(partition_canvas, net.source, sink)
            new_line = Line(net.source, sink, line_id)
        else:
            new_line = Line(net.source, sink, -1)
        net.lines.append(new_line)
        unique_line_list.append(new_line)


def draw_line(partition_canvas, source: Cell, sink: Cell):
    """
    Draws a line between two placed cells
    """

    # Get line coordinates
    source_centre = source.site.canvas_centre
    source_x = source_centre[0]
    source_y = source_centre[1]
    sink_centre = sink.site.canvas_centre
    sink_x = sink_centre[0]
    sink_y = sink_centre[1]

    # Draw the line
    line_id = partition_canvas.create_line(source_x, source_y, sink_x, sink_y, fill='red', width=0.01)

    return line_id


def redraw_line(partition_canvas, line: Line):
    """
    Redraw an existing line.
    Used when the line's source or sink has moved since last draw.
    """
    source_centre = line.source.site.canvas_centre
    source_x = source_centre[0]
    source_y = source_centre[1]
    sink_centre = line.sink.site.canvas_centre
    sink_x = sink_centre[0]
    sink_y = sink_centre[1]
    partition_canvas.coords(line.canvas_id, source_x, source_y, sink_x, sink_y)


def key_handler(partition_canvas, event):
    """
    Accepts a key event and makes an appropriate decision.
    :param partition_canvas: Tkinter canvas
    :param event: Key event
    :return: void
    """

    e_char = event.char
    if e_char == '0':
        partition_to_completion(partition_canvas)
    elif str.isdigit(e_char):
        multistep(partition_canvas, int(e_char))
    else:
        pass


def partition_to_completion(partition_canvas):
    """
    Execute Simulated Annealing to completion.
    :param partition_canvas: Tkinter canvas
    :return: void
    """

    start = time.time()  # Record time taken for full placement
    while not partitioning_done:
        step(partition_canvas)
    end = time.time()
    elapsed = end - start
    print("Took " + str(elapsed) + "s")


def multistep(partition_canvas, n):
    """
    Perform multiple iterations of partitioning
    :param partition_canvas: Tkinter canvas
    :param n: Number of iterations
    :return: void
    """

    # Redraw lines on GUI to reflect current state of partitioning
    if partition_canvas is not None:
        redraw_all_lines(partition_canvas)


def step(partition_canvas):
    pass


def redraw_all_lines(partition_canvas: Canvas):
    """
    Redraw all of the lines in the GUI from scratch.
    """
    global unique_line_list

    for line in unique_line_list:
        redraw_line(partition_canvas, line)


def move(cell: Cell, x: int, y: int, delta: float):
    """
    Move a cell to an empty site
    """
    global current_cost
    global sa_temp

    # Move the cell
    old_site = cell.site
    cell.site = partition_grid[y][x]
    old_site.isOccupied = False
    old_site.occupant = None
    cell.site.isOccupied = True
    cell.site.occupant = cell

    # Update total cost
    current_cost += delta


def swap(cell_a: Cell, cell_b: Cell, delta: float):
    """
    Swap the locations (occupied sites) of two cells
    """
    global current_cost

    # Swap the cells
    temp_site = cell_a.site
    cell_a.site = cell_b.site
    cell_b.site = temp_site
    cell_a.site.occupant = cell_a
    cell_b.site.occupant = cell_b

    # Update total cost
    current_cost += delta


def create_partition_grid(routing_file) -> list[list[Site]]:
    """
    Create the 2D placement grid
    :param routing_file: Path to the file with circuit info
    :return: list[list[Cell]] - Routing grid
    """
    global num_cells_to_place
    global num_cell_connections
    global grid_width
    global grid_height
    global cell_dict
    global net_dict
    global partition_grid
    global range_window_half_length
    global half_grid_max_dim

    grid_line = routing_file.readline()

    # Create the routing grid
    num_cells_to_place = int(grid_line.split(' ')[0])
    num_cell_connections = int(grid_line.split(' ')[1])
    grid_height = ceil(num_cells_to_place/2)
    partition_grid = []
    # Create grid in column-major order
    for _ in range(grid_height):
        partition_grid.append([])
    # Populate grid with sites
    for cell_y, row in enumerate(partition_grid):
        for cell_x in range(grid_width):
            row.append(Site(x=cell_x, y=cell_y))

    # Keep a cell dictionary
    for cell_id in range(num_cells_to_place):
        cell_dict[cell_id] = Cell(cell_id)

    # Create nets
    new_net_id = -1
    for line_num, line in enumerate(routing_file):
        net_tokens = line.split(' ')
        new_net_id += 1

        if len(net_tokens) < 2:
            # Invalid line
            new_net_id += -1
            continue

        num_cells_in_net = int(net_tokens[0])

        # Create new net
        new_net = Net(line_num, num_cells_in_net)
        net_dict[line_num] = new_net

        # Add cells to net
        source_id = int(net_tokens[1])  # Add source cell first
        source_cell = cell_dict[source_id]
        new_net.source = source_cell
        source_cell.nets.append(new_net)
        for sink_idx in range(2, num_cells_in_net+1):
            if net_tokens[sink_idx] == '\n' or net_tokens[sink_idx] == '':
                continue
            else:
                sink_id = int(net_tokens[sink_idx])
                sink_cell = cell_dict[sink_id]
                new_net.sinks.append(sink_cell)
                sink_cell.nets.append(new_net)

    return partition_grid

