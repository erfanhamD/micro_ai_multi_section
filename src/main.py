import numpy as np
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import utils

if __name__=="__main__":
    # Generate a random polygon
    corners = utils.load_section_geometry("section_1.csv")
    polygon = Polygon(corners)
    vertices = polygon.boundary.coords
    centeroid = polygon.centroid
    edges = [LineString(vertices[k:k+2]) for k in range(len(vertices) - 1)]
    projections = [edges[i].interpolate(edges[i].project(centeroid)) for i in range(len(edges))]
    print(edges)
    print(projections)
    plt.figure()
    # plot the polygon
    plt.scatter(*polygon.exterior.xy, color = 'k')
    plt.scatter(*centeroid.xy, color  = 'r')
    for i in range(len(projections)):
        plt.scatter(*projections[i].xy, color = 'orange')
    # centers = utils.calculate_center(corners)
    # side_number = corners.shape[0]-1
    # projections = []
    # for i in range(side_number):
    #     projections.append(utils.project_center_to_side(corners[[i, i+1], :], centers))
    # plt.figure()
    # plt.scatter(corners[:, 0], corners[:, 1])
    # plt.scatter(centers[0], centers[1], color = 'r')
    # plt.scatter(projections[:][0], projections[:][1], color = 'g')
    plt.show()
