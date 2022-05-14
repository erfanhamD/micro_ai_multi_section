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
    corners = utils.load_section_geometry("section_1.csv")
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
            chunk.transform_from_mosoli_to_chunk()
            chunks.append(chunk)
            polygon = chunk.polygon()
            plot_coords(ax, polygon.exterior)
            patch = PolygonPatch(polygon, facecolor=chunk_color_list[chunk.tri_type()], edgecolor=color_isvalid(chunk), alpha=0.5, zorder=2)
            ax.add_patch(patch)
            polygon_centroid_coordinate = polygon.centroid.coords[0]
            plt.text(polygon_centroid_coordinate[0], polygon_centroid_coordinate[1], str(chunk_label))
            chunk_label += 1
    plt.show()
    # chunk.chunk_lift()
    # utils.plot_chunk(polygon, vertices, corners, centeroid, projections)
    # model_address = '/Users/venus/AI_lift/multi_section/model/model_state_dict_3Apr_mm'
    # model = inference.Lift_base_network()
    # model.load_state_dict(torch.load(model_address))
    # input_params = np.loadtxt(os.path.join(CONF.DATA_DIR, "input_params/input1.csv"), delimiter=',')
    # Re = input_params[0]
    # kappa = input_params[1]
    # x_limit = input_params[2]
    # y_limit = input_params[3]
    # grid_size = int(input_params[4])
    # Data = chunk.inference_data(Re, kappa, x_limit, y_limit, grid_size)
    # lower_tri_mask, upper_tri_mask = utils.mask_section(Data[:, -2:])
    # Data = inference.preprocess(Data)
    # Cl_map = inference.inference(Data, model)
    # Cl_lower = Cl_map[lower_tri_mask]
    # Cl_upper = Cl_map[upper_tri_mask]
    # print("lsfajlfjalfj")
