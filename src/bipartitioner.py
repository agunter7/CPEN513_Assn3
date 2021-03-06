"""
Solution to UBC CPEN 513 Assignment 3.
Implements branch-and-bound based bi-partitioning.
Uses Tkinter for GUI.
"""

import os
import random
import time
from tkinter import *
from math import exp, sqrt, ceil
from copy import deepcopy
from queue import PriorityQueue

# Constants
FILE_DIR = "../benchmarks/"

# Global variables
file_name = ""  # Name of benchmark file to open
num_nodes_to_part = 0  # Number of nodes in the graph
num_node_connections = 0  # Number of edges in the graph
grid_width = 2  # Width of the partition grid
grid_height = 0  # Height of the partition grid
node_dict = {}  # Dictionary of all nodes, key is node ID
net_dict = {}  # Dictionary of all nets, key is net ID
partition_grid = []  # 2D list of sites for partitioned nodes
partitioning_done = False  # Is the partition complete?
root = None  # Tkinter root
unique_line_list = []  # List of unique lines across multiple nets
solution_is_optimal = True  # Is the final solution (known to be) optimal? Default true, prove false

# Partitioning variables
best_partition = None  # The best partition observed so far
partition_pq = PriorityQueue()  # Priority queue of partial solution partitions
current_tree_depth = 0  # Current depth of the decision tree being explored
node_id_pq = PriorityQueue()  # Priority queue of the nodes to be placed in a partition
max_nodes_per_half = 0  # Maximum number of nodes that can be placed in half of a bi-partition


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
        self.text_id = None  # ID of the Tkinter text inside this site


class Node:
    """
    A single node
    """
    def __init__(self, node_id):
        self.id = node_id  # Identifier
        self.isPlaced = False  # Has this node been placed into a site?
        self.isSource = False  # Is this node a source node?
        self.site = None  # Reference to the site this node occupies
        self.nets = []  # Nets this node is a part of
        self.family = []  # Extended Neighbourhood: Neighbours + neighbours of neighbours, in original graph


class NodeCluster:
    """
    A collection of nodes
    """
    def __init__(self):
        self.nodes = []  # Nodes in the cluster
        self.family = []  # Concatenation of nodes' families

    def add_node(self, node):
        """
        Add a node to this cluster
        :param node: Node to add
        """
        self.nodes.append(node)
        for new_family_member in node.family:
            self.family.append(new_family_member)


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
    id_counter = 0  # Counter for all partition IDs

    def __init__(self):
        self.left = []  # Left half of partition
        self.right = []  # Right half of partition
        self.cost = 0  # Cost of this partition
        self.id = Partition.get_id()  # ID of this partition
        self.parent_id = 0  # ID of the parent partition
        Partition.id_counter += 1

    def __lt__(self, other) -> bool:
        """
        Overwrite Python dunder for '<' operator
        :param other: Another Partition
        :return: bool for ordering
        """
        return self.cost > other.cost  # Want to reverse the order for priority queue

    @staticmethod
    def get_id():
        """
        Get an ID from class counter
        """
        temp = Partition.id_counter
        Partition.id_counter += 1
        return temp

    def copy(self):
        """
        Make a copy of this partition
        """
        copy_partition = deepcopy(self)
        copy_partition.parent_id = copy_partition.id
        copy_partition.id = Partition.get_id()
        return copy_partition

    def calculate_cost(self):
        """
        Calculate the cost of this partition.
        Updates self.cost
        """
        local_partition_cost = 0

        for net in net_dict.values():
            net_on_left = False
            net_on_right = False
            if net.source.id in self.left:
                net_on_left = True
            elif net.source.id in self.right:
                net_on_right = True
            for node in net.sinks:
                if node.id in self.left:
                    net_on_left = True
                if node.id in self.right:
                    net_on_right = True
            if net_on_left and net_on_right:
                # Net is split
                local_partition_cost += 1

        self.cost = local_partition_cost

    def add_node(self, node: Node, add_left=True):
        """
        Add a node to this partition
        :param node: Node to place
        :param add_left: Add the node to the left partition? Else, add right.
        """
        # Check if node will split any nets
        for net in node.nets:
            net_on_left = False
            net_on_right = False
            if net.source.id in self.left:
                net_on_left = True
            elif net.source.id in self.right:
                net_on_right = True
            for net_node in net.sinks:
                if net_node.id in self.left:
                    net_on_left = True
                if net_node.id in self.right:
                    net_on_right = True
            if net_on_left and net_on_right:
                # Net is already split
                continue
            elif net_on_right:
                if add_left:
                    # Net will be split
                    self.cost += 1
            elif net_on_left:
                if not add_left:
                    # Net will be split
                    self.cost += 1

        # Add the node to the partition
        if add_left:
            self.left.append(node.id)
        else:
            self.right.append(node.id)

    def is_balanced(self) -> bool:
        """
        Is this partition balanced?
        Defined within context of a full solution, don't care about partial solutions.
        E.g. For 100 nodes to place, a 10 to 2 partition with 88 remaining is fine
        :return: boolean
        """
        global max_nodes_per_half

        if len(self.left) > max_nodes_per_half or len(self.right) > max_nodes_per_half:
            return False
        else:
            return True


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

    
def partition_algorithm(f_name: str):
    """
    Perform partitioning with a GUI.
    :param f_name: Name of file to open
    :return: void
    """
    global FILE_DIR
    global file_name
    global num_nodes_to_part
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


def partition_to_completion(partition_canvas):
    """
    Execute branch-and-bound partitioning to completion.
    :param partition_canvas: Tkinter canvas
    :return: void
    """

    start = time.time()  # Record time taken for full partition
    while not partitioning_done:
        step_start = time.time()
        step()
        step_end = time.time()
        step_elapsed = step_end - step_start
        print("Previous step took " + str(step_elapsed) + "s")
    end = time.time()
    elapsed = end - start
    print("Took " + str(elapsed) + "s")

    # Update GUI with final partition
    replace_partition(partition_canvas)


def step():
    """
    Explore one layer of the partitioning decision tree
    """
    global best_partition
    global node_id_pq
    global partition_pq
    global partitioning_done
    global current_tree_depth
    global solution_is_optimal

    # Check if algorithm is complete
    if node_id_pq.empty() or partition_pq.empty():
        partitioning_done = True
        # Update best partition
        while not partition_pq.empty():
            _, partition = partition_pq.get()
            if partition.cost < best_partition.cost:
                best_partition = partition
        print("Final partition:")
        print("Left" + str(best_partition.left))
        print("Right" + str(best_partition.right))
        if solution_is_optimal:
            print("Optimal solution achieved.")
        else:
            print("Heuristic solution achieved.")
        print("Final cost: " + str(best_partition.cost))
        return

    print("Partitioning " + str(partition_pq.qsize()) + " nodes at tree depth of " + str(current_tree_depth))

    _, next_node_id = node_id_pq.get()
    next_node = node_dict[next_node_id]
    current_depth_partitions = PriorityQueue()

    while not partition_pq.empty():
        # Create left and right partitions
        _, new_partition_left = partition_pq.get()
        new_partition_right = new_partition_left.copy()
        new_partition_left.add_node(next_node, add_left=True)
        new_partition_right.add_node(next_node, add_left=False)
        # Check if new partitions are valid, prune by cost
        if new_partition_left.cost < best_partition.cost and new_partition_left.is_balanced():
            current_depth_partitions.put((-1*new_partition_left.cost, new_partition_left))
        if new_partition_right.cost < best_partition.cost and new_partition_right.is_balanced():
            current_depth_partitions.put((-1*new_partition_right.cost, new_partition_right))

    partition_pq = current_depth_partitions
    # Forcibly prune partition tree to a size (2^20) that will yield manageable runtime
    # PQ sorted by cost, will remove max cost items first
    while partition_pq.qsize() > 1048576:
        solution_is_optimal = False
        _, _ = partition_pq.get()

    current_tree_depth += 1


def initial_partition(partition_canvas):
    """
    Perform an initial partition prior to branch-and-bound
    :param partition_canvas: Tkinter canvas
    """
    global partition_grid
    global num_nodes_to_part
    global best_partition

    # Setup list and priority queue of nodes
    node_id_list = []
    for node_id in node_dict.keys():
        node_id_list.append(node_id)  # For random partition search
        node_id_pq.put((-1*len(node_dict[node_id].nets), node_id))  # For later branch-and-bound processing

    # Sample random bi-partitioning solutions
    rand_start = time.time()
    best_partition = Partition()
    best_random_cost = float("inf")
    for _ in range(50000):
        new_partition = Partition()
        random.shuffle(node_id_list)
        for node_idx, node_id in enumerate(node_id_list):
            # Place nodes randomly into partition
            if node_idx % 2 == 0:
                new_partition.left.append(node_id)
            else:
                new_partition.right.append(node_id)
        new_partition.calculate_cost()
        if new_partition.cost < best_random_cost:
            best_random_cost = new_partition.cost
            best_partition = new_partition
    rand_end = time.time()
    print("Initial random cost: " + str(best_partition.cost))
    print("Random search took " + str(rand_end-rand_start) + "s")

    # Cluster nodes intelligently
    # Get family of each node
    intel_start = time.time()
    use_small_families = False
    for root_node in node_dict.values():
        for immediate_net in root_node.nets:
            root_node.family.append(immediate_net.source.id)
            for immediate_sink in immediate_net.sinks:
                root_node.family.append(immediate_sink.id)
                for distant_net in immediate_sink.nets:
                    root_node.family.append(distant_net.source.id)
                    for distant_sink in distant_net.sinks:
                        root_node.family.append(distant_sink)
        if len(root_node.family) > 1500:
            # Runtime will be quite large, use neighbourhood instead of extended neighbourhood as node family
            use_small_families = True
    if use_small_families:
        print("Using regular neighbourhood")
        for root_node in node_dict.values():
            root_node.family = []
            for immediate_net in root_node.nets:
                root_node.family.append(immediate_net.source.id)
                for immediate_sink in immediate_net.sinks:
                    root_node.family.append(immediate_sink.id)

    # Create clusters that only contain a single node as a starting point
    cluster_list = []
    for candidate_node in node_dict.values():
        new_cluster = NodeCluster()
        new_cluster.nodes.append(candidate_node)
        new_cluster.family = candidate_node.family
        cluster_list.append(new_cluster)

    # Iteratively group clusters together until only 2 clusters remain
    holdout_clusters = []
    while len(cluster_list) > 2:
        candidate_list = []
        while cluster_list:
            candidate_list.append(cluster_list.pop())
        if len(candidate_list) % 2 == 1:
            # Odd number of candidates
            # Holdout smallest cluster (by family size) until end of cluster
            smallest_cluster = None
            smallest_family_size = float("inf")
            for holdout_candidate in candidate_list:
                if len(holdout_candidate.family) < smallest_family_size:
                    smallest_family_size = len(holdout_candidate.family)
                    smallest_cluster = holdout_candidate
            holdout_clusters.append(smallest_cluster)
            candidate_list.remove(smallest_cluster)
        if len(candidate_list) == 2:
            # Clustering is done
            cluster_list = candidate_list
            break
        # Merge pairs of remaining clusters
        while candidate_list:
            # Pick a starting cluster by largest family
            starting_cluster = None
            biggest_family = -1
            for candidate_cluster in candidate_list:
                if len(candidate_cluster.family) > biggest_family:
                    starting_cluster = candidate_cluster
                    biggest_family = len(candidate_cluster.family)
            candidate_list.remove(starting_cluster)  # Starting cluster no longer eligible
            # Find the cluster with the most familial overlap with the starting cluster
            best_partner_cluster = None
            best_overlap = -1
            for candidate_cluster in candidate_list:
                temp_overlap = count_family_overlap(starting_cluster, candidate_cluster)
                if temp_overlap > best_overlap:
                    best_overlap = temp_overlap
                    best_partner_cluster = candidate_cluster
            # Merge the two clusters
            for new_node in best_partner_cluster.nodes:
                starting_cluster.add_node(new_node)
            candidate_list.remove(best_partner_cluster)  # Partner cluster no longer eligible
            cluster_list.append(starting_cluster)  # Add the merged clusters back to the original list as one cluster
    if len(cluster_list) != 2:
        print("Clustering error: number of clusters is " + str(len(cluster_list)))
        exit()
    # Apportion nodes from holdout clusters to one of the two remaining clusters
    for holdout_cluster in holdout_clusters:
        for holdout_node in holdout_cluster.nodes:
            left_overlap = count_family_overlap(holdout_node, cluster_list[0])
            right_overlap = count_family_overlap(holdout_node, cluster_list[1])
            if left_overlap > right_overlap:
                cluster_list[0].add_node(holdout_node)
            else:
                cluster_list[1].add_node(holdout_node)

    # Legalize clusters
    while len(cluster_list[0].nodes) - len(cluster_list[1].nodes) > 1:
        # Move node with smallest family from bigger cluster into smaller cluster
        emigrant_node = None
        smallest_family_size = float("inf")
        for surplus_node in cluster_list[0].nodes:
            if len(surplus_node.family) < smallest_family_size:
                emigrant_node = surplus_node
                smallest_family_size = len(surplus_node.family)
        cluster_list[1].add_node(emigrant_node)
        cluster_list[0].nodes.remove(emigrant_node)  # TODO: Note that cluster family should be reduced
    while len(cluster_list[1].nodes) - len(cluster_list[0].nodes) > 1:
        # Move node with smallest family from bigger cluster into smaller cluster
        emigrant_node = None
        smallest_family_size = float("inf")
        for surplus_node in cluster_list[1].nodes:
            if len(surplus_node.family) < smallest_family_size:
                emigrant_node = surplus_node
                smallest_family_size = len(surplus_node.family)
        cluster_list[0].add_node(emigrant_node)
        cluster_list[1].nodes.remove(emigrant_node)  # TODO: Note that cluster family should be reduced

    # Set the two clusters as a partition
    intelligent_partition = Partition()
    for left_node in cluster_list[0].nodes:
        intelligent_partition.left.append(left_node.id)
    for right_node in cluster_list[1].nodes:
        intelligent_partition.right.append(right_node.id)
    intelligent_partition.calculate_cost()
    intel_end = time.time()

    # Decide on best partition to carry forward as an initial partition
    if intelligent_partition.cost < best_partition.cost:
        best_partition = intelligent_partition
    print("Initial intelligent cost: " + str(intelligent_partition.cost))
    print("Intelligent partition took " + str(intel_end - intel_start) + "s")

    # GUI update for placements
    place_partition(partition_canvas, best_partition)

    # Draw nets
    for net in net_dict.values():
        # Draw net on canvas
        if partition_canvas is not None:
            draw_net(partition_canvas, net)

    partition_pq.put((0, Partition()))  # A blank partition to start branch-and-bound from


def count_family_overlap(cluster1, cluster2) -> int:
    """
    Count the overlap between two clusters/nodes.
    :param cluster1: A Node or Cluster object
    :param cluster2: A Node or Cluster object
    :return: The overlap amount
    """
    overlap = 0
    for member1_id in cluster1.family:
        for member2_id in cluster2.family:
            if member1_id == member2_id:
                overlap += 1
    return overlap


def place_partition(partition_canvas: Canvas, partition: Partition):
    """
    Place a partition in the partition grid for visualization purposes
    :param partition_canvas: Tkinter canvas
    :param partition: The Partition to place
    """
    global partition_grid

    # Check that all nodes are included in the partition
    for node_id in node_dict.keys():
        if node_id not in partition.left and node_id not in partition.right:
            print("Orphan node: ")
            print(node_id)
            print(partition.left)
            print(partition.right)

    # Place left half of partition
    x = 0
    partition.left.sort()
    for y, node_id in enumerate(partition.left):
        node_to_place = node_dict[node_id]
        place_node(partition_canvas, node_to_place, x, y)
    x = 1
    # Place right half of partition
    partition.right.sort()
    for y, node_id in enumerate(partition.right):
        node_to_place = node_dict[node_id]
        place_node(partition_canvas, node_to_place, x, y)


def place_node(partition_canvas: Canvas, node, x, y):
    """
    Place a single node in the partition grid
    :param partition_canvas: Tkinter canvas
    :param node: The Node to place
    :param x: The x position in the grid to place the node
    :param y: The y position in the grid to place the node
    """
    global partition_grid

    partition_site = partition_grid[y][x]
    partition_site.occupant = node
    node.site = partition_site
    partition_site.isOccupied = True
    node.isPlaced = True
    site_rect_coords = partition_canvas.coords(partition_site.canvas_id)
    text_x = (site_rect_coords[0] + site_rect_coords[2]) / 2
    text_y = (site_rect_coords[1] + site_rect_coords[3]) / 2
    # Check if node is a source in a split net
    node_is_split_source = False
    for net in net_dict.values():
        if net.source == node:
            # Node is source in this net
            # Check if this net is split
            net_on_left = False
            net_on_right = False
            if net.source.id in best_partition.left:
                net_on_left = True
            if net.source.id in best_partition.right:
                net_on_right = True
            for sink_node in net.sinks:
                if sink_node.id in best_partition.left:
                    net_on_left = True
                if sink_node.id in best_partition.right:
                    net_on_right = True
            if net_on_left and net_on_right:
                node_is_split_source = True
    if node_is_split_source:
        text_colour = 'blue'
    else:
        text_colour = 'black'
    partition_site.text_id = partition_canvas.create_text(text_x, text_y, font=("arial", 10),
                                                          text=str(node.id), fill=text_colour)


def remove_node(partition_canvas: Canvas, node):
    """
    Remove a node from its site in the partition grid
    :param partition_canvas: Tkinter canvas
    :param node: The node to remove
    """
    global partition_grid

    node.isPlaced = False
    node.site.isOccupied = False
    node.site.occupant = None
    partition_canvas.delete(node.site.text_id)
    node.site = None


def replace_partition(partition_canvas: Canvas):
    """
    Place the best partition from scratch
    :param partition_canvas: Tkinter canvas
    """
    global best_partition

    # Remove all nodes from the grid
    for node in node_dict.values():
        remove_node(partition_canvas, node)

    place_partition(partition_canvas, best_partition)

    redraw_all_lines(partition_canvas)


def draw_net(partition_canvas, net):
    """
    Draw a net on the canvas from scratch
    :param partition_canvas: Tkinter canvas
    :param net: The Net to draw
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
    :param partition_canvas: Tkinter canvas
    :param source: source Node
    :param sink: sink Node
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
    :param partition_canvas: Tkinter canvas
    :param line: The Line to draw
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
    else:
        pass


def redraw_all_lines(partition_canvas: Canvas):
    """
    Redraw all of the lines in the GUI from scratch.
    :param partition_canvas: Tkinter canvas
    """
    global unique_line_list

    for line in unique_line_list:
        redraw_line(partition_canvas, line)


def create_partition_grid(routing_file) -> list[list[Site]]:
    """
    Create the 2D partition grid
    :param routing_file: Path to the file with circuit info
    :return: list[list[Node]] - Routing grid
    """
    global num_nodes_to_part
    global num_node_connections
    global grid_width
    global grid_height
    global node_dict
    global net_dict
    global partition_grid
    global max_nodes_per_half

    grid_line = routing_file.readline()

    # Create the routing grid
    num_nodes_to_part = int(grid_line.split(' ')[0])
    num_node_connections = int(grid_line.split(' ')[1])
    grid_height = ceil(num_nodes_to_part / 2)
    max_nodes_per_half = grid_height
    partition_grid = []
    # Create grid in column-major order
    for _ in range(grid_height):
        partition_grid.append([])
    # Populate grid with sites
    for node_y, row in enumerate(partition_grid):
        for node_x in range(grid_width):
            row.append(Site(x=node_x, y=node_y))

    # Keep a node dictionary
    for node_id in range(num_nodes_to_part):
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
        source_node.isSource = True
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
