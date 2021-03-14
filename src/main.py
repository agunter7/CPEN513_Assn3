import bipartitioner

# File name for interactive program (with GUI). Edit this to change the netlist being annealed.
USER_FILE_NAME = "ugly8.txt"

# file_name - nodes - connections
# cc 62 42, cm82a 12 9, cm138a 24 16, cm150a 36 35, cm162a 37 32,
# con1 14 12, twocm 70 69, ugly8 8 8, ugly16 16 16, z4ml 19 15


def main():
    """
    Main function for running bi-partitioning
    """
    bipartitioner.partition_algorithm(USER_FILE_NAME)


if __name__ == "__main__":
    main()
