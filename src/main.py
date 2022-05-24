import os
import numpy as np
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
from config import CONF
import utils
import inference
from chunk import Chunk
from figures import BLUE, RED, SIZE, set_limits, plot_coords, color_isvalid
from descartes.patch import PolygonPatch
import torch
import torch.nn as nn


if __name__ == "__main__":

    corners = utils.load_section_geometry("section_3.csv")
    centeroid_ = Polygon(corners).centroid
    corners = corners - centeroid_
    polygon, vertices, centeroid = utils.polygon_from_corners(corners)
    
    edges = [LineString(vertices[k:k+2]) for k in range(len(vertices) - 1)]
    projections = [edges[i].interpolate(edges[i].project(centeroid)) for i in range(len(edges))]
    chunks = []
    chunk_color_list = [BLUE, RED]
    fig = plt.figure(1, figsize=SIZE, dpi=90)
    ax = fig.add_subplot(111)
    chunk_label = 0
    for idx, edge in enumerate(edges):
        for jdx, vertex in enumerate(edge.boundary):
            chunk = Chunk(centeroid, vertex, projections[idx])
            # chunk.transform_from_mosoli_to_chunk()
            chunks.append(chunk)
            polygon = chunk.polygon()
            plot_coords(ax, polygon.exterior)
            patch = PolygonPatch(polygon, facecolor=chunk_color_list[chunk.tri_type()], edgecolor=color_isvalid(chunk), alpha=0.5, zorder=2)
            ax.add_patch(patch)
            polygon_centroid_coordinate = polygon.centroid.coords[0]
            plt.text(polygon_centroid_coordinate[0], polygon_centroid_coordinate[1], str(chunk_label))
            chunk_label += 1
    # plt.show()
    #chunks[7].chunk_lift()
    # plt.show()
    # chunks[1].chunk_lift()
    for ch in chunks:
        ch.chunk_lift()
    plt.show()