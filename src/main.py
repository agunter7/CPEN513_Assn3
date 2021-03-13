import cProfile
import pstats
import bipartitioner

# Experimental grid search parameters (do not alter)
FILE_NAMES = ["cc.txt", "cm82a.txt", "cm138a.txt", "cm150a.txt", "cm162a.txt", "con1.txt", "twocm.txt", "ugly8.txt",
              "ugly16.txt", "z4ml.txt"]

# File name for interactive program (with GUI). Edit this to change the netlist being annealed.
USER_FILE_NAME = "cm138a.txt"

# Notes
# cc 62 42, cm82a 12 9, cm138a 24 16, cm150a 36 35, cm162a 37 32,
# con1 14 12, twocm 70 69, ugly8 8 8, ugly16 16 16, z4ml 19 15


def main():
    """
    Main function for running annealing experiments
    """
    experimental_mode = False

    if experimental_mode:
        pass
        print("Experimental mode not in use.")
    else:
        bipartitioner.partition_algorithm(USER_FILE_NAME)


if __name__ == "__main__":
    main()
