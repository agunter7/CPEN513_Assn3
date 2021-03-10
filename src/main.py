import cProfile
import pstats
import sim_anneal

# Experimental grid search parameters (do not alter)
FILE_NAMES = ["cc.txt", "cm82a.txt", "cm138a.txt", "cm150a.txt", "cm162a.txt", "con1.txt", "twocm.txt", "ugly8.txt",
              "ugly16.txt", "z4ml.txt"]

# File name for interactive program (with GUI). Edit this to change the netlist being annealed.
USER_FILE_NAME = "cm82a.txt"


def main():
    """
    Main function for running annealing experiments
    """
    experimental_mode = False

    if experimental_mode:
        pass
        print("Experimental mode not in use.")
    else:
        sim_anneal.partition_algorithm(USER_FILE_NAME)


if __name__ == "__main__":
    main()
