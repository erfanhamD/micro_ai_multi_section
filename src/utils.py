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