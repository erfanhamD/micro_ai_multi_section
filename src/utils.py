import os
import sys
import numpy as np
import argparse
from config import CONF

DATA_DIR = CONF.DATA_DIR

def polygon_generator(n_corners=4):
    corners = np.random.randint(0, 10, (n_corners, 2))
    return corners

def calculate_center(corners):
    center = np.mean(corners, axis=0)
    return center

def load_section_geometry(section_file):
    """
    Loads the section geometry from a file.
    """
    section_addr = os.path.join(DATA_DIR, section_file)
    corners = np.loadtxt(section_addr, dtype=np.float32, delimiter=',')
    return corners

def project_center_to_side(corners, center):
    corner_1 = corners[0]
    corner_2 = corners[1]
    side = corner_2 - corner_1
    side_norm = np.linalg.norm(side)
    projection_line = center - corner_1
    projection = (np.dot(side, projection_line)/side_norm**2) * side + corner_1
    return projection
