import os
import sys
import numpy as np
import argparse
from config import CONF
import shapely
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
import inference
import torch
import torch.nn as nn

DATA_DIR = CONF.DATA_DIR

def polygon_generator(n_corners=4):
    """
    generates a random polygon with n_corners corners.
    """
    corners = np.random.randint(0, 10, (n_corners, 2))
    return corners

def calculate_center(corners):
    """
   Calculates the center of the polygon. 
    """
    center = np.mean(corners, axis=0)
    return center

def load_section_geometry(section_file):
    """
    Loads the section geometry from a file.
    """
    section_addr = os.path.join(DATA_DIR, f"section/{section_file}")
    corners = np.loadtxt(section_addr, dtype=np.float32, delimiter=',')
    
    return corners

def project_center_to_side(corners, center):
    """
    Projects the center of the polygon to each of the sides. 
    """
    corner_1 = corners[0]
    corner_2 = corners[1]
    side = corner_2 - corner_1
    side_norm = np.linalg.norm(side)
    projection_line = center - corner_1
    projection = (np.dot(side, projection_line)/side_norm**2) * side + corner_1
    return projection

def compute_aspect_ratio(projection, centeroid, vertex, tri_type):
    """
    Calculates the AR based on the triangle type. 
    """
    x_dist = projection.distance(vertex)
    y_dist = centeroid.distance(projection)
    if tri_type:
        return x_dist/y_dist
    else:
        return y_dist/x_dist

def line_angle_calc(line):
    """
    Returns the type of the triangle.
    """
    horizon_line = [1, 0]
    line_origin = np.array(line)[1]-np.array(line)[0]
    dot_product = np.dot(horizon_line, line_origin)
    line_norm = np.linalg.norm(np.array(line)[1]-np.array(line)[0])
    angle = np.arccos(dot_product/line_norm)
    return angle

def plot_line(line):
    """
    Plots the line.
    """
    plt.plot(*line.xy)

def tri_type(om, oc):
    """
    Returns the type of the triangle.
    """
    if line_angle_calc(om)>line_angle_calc(oc):
        return 1
    else:
        return 0

def polygon_from_corners(corners):
    """
    Creates a shapely polygon from the corners.
    """
    polygon = Polygon(corners)
    vertices = polygon.boundary.coords
    centeroid = polygon.centroid
    return polygon, vertices, centeroid

def plot_chunk(polygon, vertices, corners, centeroid, projections):
    """
    Plots the chunks and the cross sections. 
    """
    centeroid_corner_line = [LineString([vertices[i], centeroid]) for i in range(len(vertices)-1)]
    plt.scatter(*polygon.exterior.xy, color = 'k')
    plt.plot(corners[:,0], corners[:,1], '--k')
    plt.scatter(*centeroid.xy, color  = 'r', linewidths=30, alpha = 0.3)
    for i in range(len(projections)):
        plt.scatter(*projections[i].xy, color = 'orange')
        plt.plot(*centeroid_corner_line[i].xy, '--b')
    plt.gca().set_aspect('equal')
    plt.show()

def mask_section(Data, limits):
    # find the diameter of the grid
    [x_limit, y_limit] = limits
    min_point = np.min(Data, axis=0)
    max_point = np.max(Data, axis=0)
    line_slope =(np.max(Data[:,0]) - np.min(Data[:,0]))/(np.max(Data[:,1]) - np.min(Data[:,1]))
    line_slope = y_limit/x_limit
    point_slope = calc_slope(Data)
    mask_l = point_slope < line_slope
    mask_u = point_slope > line_slope
    upper_tri = Data[mask_u]
    lower_tri = Data[mask_l]

    # plt.scatter(upper_tri[:,0], upper_tri[:,1], color = 'r')
    # plt.scatter(lower_tri[:,0], lower_tri[:,1], color = 'b')
    # plt.figure()
    # plt.scatter(Data[:,0], Data[:,1], color = 'k')
    # plt.plot([min_point[0], max_point[0]], [min_point[1], max_point[1]], '--k')
    # plt.show()
    return mask_l, mask_u

def calc_slope(Point):
    """
    Calculates the slope of the line.
    """
    slope = Point[:, 0]/Point[:, 1]
    return slope
