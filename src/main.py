import numpy as np
import matplotlib.pyplot as plt
import utils

if __name__=="__main__":
    # Generate a random polygon
    corners = utils.load_section_geometry("section_1.csv")
    centers = utils.calculate_center(corners)
    plt.figure()
    plt.scatter(corners[:, 0], corners[:, 1])
    plt.scatter(centers[0], centers[1], color = 'r')
    plt.show()
    centers = utils.calculate_center(corners)