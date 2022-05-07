import numpy as np
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import utils
import inference
from chunk import Chunk
from figures import BLUE, RED, SIZE, set_limits, plot_coords, color_isvalid
from descartes.patch import PolygonPatch
import torch
import torch.nn as nn


if __name__ == "__main__":
    corners = utils.load_section_geometry("section_0.csv")
    polygon, vertices, centeroid = utils.polygon_from_corners(corners)
    centeroid_corner_line = [LineString([vertices[i], centeroid]) for i in range(len(vertices)-1)]
    edges = [LineString(vertices[k:k+2]) for k in range(len(vertices) - 1)]
    projections = [edges[i].interpolate(edges[i].project(centeroid)) for i in range(len(edges))]
    chunks = []
    chunk_color_list = [BLUE, RED]
    fig = plt.figure(1, figsize=SIZE, dpi=90)
    ax = fig.add_subplot(111)
    for idx, edge in enumerate(edges):
        for jdx, vertex in enumerate(edge.boundary):
            chunk = Chunk(centeroid, vertex, projections[idx])
            chunk.tri_type()
            chunks.append(chunk)
            polygon = chunk.polygon()
            plot_coords(ax, polygon.exterior)
            patch = PolygonPatch(polygon, facecolor=chunk_color_list[jdx], edgecolor=color_isvalid(polygon, valid=chunk_color_list[jdx]), alpha=0.5, zorder=2)
            ax.add_patch(patch)
    chunks[0].chunk_AR()
    plt.scatter(*polygon.exterior.xy, color = 'k')
    plt.plot(corners[:,0], corners[:,1], '--k')
    plt.scatter(*centeroid.xy, color  = 'r', linewidths=30, alpha = 0.3)
    for i in range(len(projections)):
        plt.scatter(*projections[i].xy, color = 'orange')
        plt.plot(*centeroid_corner_line[i].xy, '--b')
    plt.gca().set_aspect('equal')

    plt.show()
    input_data_addr = '/Users/venus/AI_lift/multi_section/data/z-3-50-30.csv'
    model_address = '/Users/venus/AI_lift/multi_section/model/model_state_dict_3Apr_mm'
    model = inference.Lift_base_network()
    model.load_state_dict(torch.load(model_address))
    Data = inference.preprocess(input_data_addr)
    Cl_map = inference.inference(Data, model)
    print("lsfajlfjalfj")