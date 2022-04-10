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
    AR_dict = {}
    for chunc_id in range(len(edges)):
        AR_dict[chunc_id] = utils.compute_aspect_ratio(projections[chunc_id], centeroid, edges[chunc_id])
    print(edges)
    print(projections)

    plt.figure()
    plt.scatter(*polygon.exterior.xy, color = 'k')
    plt.plot(corners[:,0], corners[:,1], '--k')
    plt.scatter(*centeroid.xy, color  = 'r')
    for i in range(len(projections)):
        plt.scatter(*projections[i].xy, color = 'orange')
    plt.show()
