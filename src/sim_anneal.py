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

# Global variables
num_nodes_to_place = 0  # Number of nodes in the circuit to be placed
num_node_connections = 0  # Number of connections to be routed, summed across all nodes/nets
grid_width = 2  # Width of the partition grid
grid_height = 0  # Height of the partition grid
node_dict = {}  # Dictionary of all nodes, key is node ID
net_dict = {}  # Dictionary of all nets, key is net ID
partition_grid = []  # 2D list of sites for partition
partitioning_done = False  # Is the partition complete?
root = None  # Tkinter root
unique_line_list = []  # List of unique lines across multiple nets

# Partitioning variables
best_partition = None


class Site:
    """
    A partition site/slot for a node to inhabit
    """
    def __init__(self, x: int, y: int):
        self.x = x  # x location
        self.y = y  # y location
        self.canvas_id = -1  # ID of corresponding rectangle in Tkinter canvas
        self.canvas_centre = (-1, -1)  # Geometric centre of Tkinter rectangle
        self.isOccupied = False  # Is the site occupied by a node?
        self.occupant = None  # Reference to occupant node
        pass


class Node:
    """
    A single node
    """
    def __init__(self, node_id):
        self.id = node_id  # Identifier
        self.isPlaced = False  # Has this node been placed into a site?
        self.site = None  # Reference to the site this node occupies
        self.nets = []  # Nets this node is a part of
        pass


class Line:
    """
    A wrapper class for Tkinter lines
    """
    def __init__(self, source: Node, sink: Node, canvas_id: int):
        self.source = source  # Reference to source node
        self.sink = sink  # Reference to sink node
        self.canvas_id = canvas_id  # Tkinter ID of line


class Partition:
    """
    A bipartite collection of nodes
    """
    def __init__(self):
        self.left = []
        self.right = []
        self.cost = 0

    def calculate_cost(self):
        local_partition_cost = 0
        net_on_left = False
        net_on_right = False

        for net in net_dict.values():
            if net.source in self.left:
                net_on_left = True
            elif net.source in self.right:
                net_on_right = True
            for node in net.sinks:
                if node in self.left:
                    net_on_left = True
                if node in self.right:
                    net_on_right = True
            if net_on_left and net_on_right:
                # Net is split
                local_partition_cost += 1

        self.cost = local_partition_cost


class Net:
    """
    A collection of nodes to be connected during routing
    """
    def __init__(self, net_id: int, num_nodes: int):
        self.id = net_id  # Identifier
        self.num_nodes = num_nodes  # Number of nodes in this net
        self.source = None  # Reference to source node
        self.sinks = []  # References to sink nodes
        self.lines = []  # References to Lines in this net
        pass


def reset_globals():
    """
    Reset (most) global variables.
    Used for repeated calls to this script from a parent script.
    """
    global num_nodes_to_place
    global num_node_connections
    global grid_width
    global grid_height
    global node_dict
    global net_dict
    global partition_grid
    global partitioning_done
    global root
    global unique_line_list

    num_nodes_to_place = 0
    num_node_connections = 0
    grid_width = 2
    grid_height = 0
    node_dict = {}
    net_dict = {}
    partition_grid = []
    partitioning_done = False
    root = None
    unique_line_list = []

    
def partition(f_name: str):
    """
    Perform anneal with a GUI.
    :param f_name: Name of file to open
    :return: void
    """
    global FILE_DIR
    global file_name
    global num_nodes_to_place
    global num_node_connections
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
        # Add a node site rectangle to the canvas
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
            # Add a node site rectangle to the canvas
            top_left_x = site_length * x
            top_left_y = site_length * y
            bottom_right_x = top_left_x + site_length
            bottom_right_y = top_left_y + site_length
            rectangle_colour = "black"
            rectangle_coords = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
            partition_canvas.create_rectangle(rectangle_coords, fill=rectangle_colour)

        x = grid_height-1
        # Add a node site rectangle to the canvas
        top_left_x = site_length * x
        top_left_y = site_length * y
        bottom_right_x = top_left_x + site_length
        bottom_right_y = top_left_y + site_length
        rectangle_colour = "white"
        rectangle_coords = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        site = partition_grid[y][1]
        site.canvas_id = partition_canvas.create_rectangle(rectangle_coords, fill=rectangle_colour)
        site.canvas_centre = ((top_left_x + bottom_right_x) / 2, (top_left_y + bottom_right_y) / 2)

    # Perform initial partition
    initial_partition(partition_canvas)

    # Event bindings and Tkinter start
    partition_canvas.focus_set()
    partition_canvas.bind("<Key>", lambda event: key_handler(partition_canvas, event))
    root.mainloop()


def initial_partition(partition_canvas):
    """
    Perform an initial partition prior to Simulated Annealing
    :param partition_canvas: Tkinter canvas
    """
    global partition_grid
    global num_nodes_to_place
    global best_partition

    best_partition = Partition()
    for node_idx, node in enumerate(node_dict.values()):
        # Place nodes randomly into partition
        if node_idx % 2 == 0:
            best_partition.left.append(node)
        else:
            best_partition.right.append(node)

    # Place left half of partition
    x = 0
    for y, node in enumerate(best_partition.left):
        place_node(node, x, y)
    x = 1
    for y, node in enumerate(best_partition.right):
        place_node(node, x, y)

    # Draw partition
    for net in net_dict.values():
        # Draw net on canvas
        if partition_canvas is not None:
            draw_net(partition_canvas, net)


def place_node(node, x, y):
    global partition_grid
    
    partition_site = partition_grid[y][x]
    partition_site.occupant = node
    node.site = partition_site
    partition_site.isOccupied = True
    node.isPlaced = True


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

    # Draw line between nodes
    for sink in net.sinks:
        if partition_canvas is not None:
            line_id = draw_line(partition_canvas, net.source, sink)
            new_line = Line(net.source, sink, line_id)
        else:
            new_line = Line(net.source, sink, -1)
        net.lines.append(new_line)
        unique_line_list.append(new_line)


def draw_line(partition_canvas, source: Node, sink: Node):
    """
    Draws a line between two placed nodes
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

    start = time.time()  # Record time taken for full partition
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


def move(node: Node, x: int, y: int):
    """
    Move a node to an empty site
    """

    # Move the node
    old_site = node.site
    node.site = partition_grid[y][x]
    old_site.isOccupied = False
    old_site.occupant = None
    node.site.isOccupied = True
    node.site.occupant = node


def swap(node_a: Node, node_b: Node):
    """
    Swap the locations (occupied sites) of two nodes
    """

    # Swap the nodes
    temp_site = node_a.site
    node_a.site = node_b.site
    node_b.site = temp_site
    node_a.site.occupant = node_a
    node_b.site.occupant = node_b


def create_partition_grid(routing_file) -> list[list[Site]]:
    """
    Create the 2D partition grid
    :param routing_file: Path to the file with circuit info
    :return: list[list[Node]] - Routing grid
    """
    global num_nodes_to_place
    global num_node_connections
    global grid_width
    global grid_height
    global node_dict
    global net_dict
    global partition_grid

    grid_line = routing_file.readline()

    # Create the routing grid
    num_nodes_to_place = int(grid_line.split(' ')[0])
    num_node_connections = int(grid_line.split(' ')[1])
    grid_height = ceil(num_nodes_to_place/2)
    partition_grid = []
    # Create grid in column-major order
    for _ in range(grid_height):
        partition_grid.append([])
    # Populate grid with sites
    for node_y, row in enumerate(partition_grid):
        for node_x in range(grid_width):
            row.append(Site(x=node_x, y=node_y))

    # Keep a node dictionary
    for node_id in range(num_nodes_to_place):
        node_dict[node_id] = Node(node_id)

    # Create nets
    new_net_id = -1
    for line_num, line in enumerate(routing_file):
        net_tokens = line.split(' ')
        new_net_id += 1

        if len(net_tokens) < 2:
            # Invalid line
            new_net_id += -1
            continue

        num_nodes_in_net = int(net_tokens[0])

        # Create new net
        new_net = Net(line_num, num_nodes_in_net)
        net_dict[line_num] = new_net

        # Add nodes to net
        source_id = int(net_tokens[1])  # Add source node first
        source_node = node_dict[source_id]
        new_net.source = source_node
        source_node.nets.append(new_net)
        for sink_idx in range(2, num_nodes_in_net+1):
            if net_tokens[sink_idx] == '\n' or net_tokens[sink_idx] == '':
                continue
            else:
                sink_id = int(net_tokens[sink_idx])
                sink_node = node_dict[sink_id]
                new_net.sinks.append(sink_node)
                sink_node.nets.append(new_net)

    return partition_grid
