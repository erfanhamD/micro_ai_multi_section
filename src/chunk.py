import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
import inference
import os
import torch
import torch.nn as nn
import utils
from config import CONF

from utils import line_angle_calc
class Chunk:
    def __init__(self, centroid, corner, projection):
        self.projection = projection
        self.centroid = centroid
        self.corner = corner
        self.oc = LineString([self.corner, self.centroid])
        self.om = LineString([self.projection, self.centroid])
    
    def mosoli_triangle(self):
        """
        Returns the mosoli triangle.
        """
        if self.tri_type():
            mosol = np.array([[0, 0, 1], [1, 0 ,1], [1, 1, 1]])
            # self.transform_from_mosoli_to_chunk(mosol)
            return mosol
        else:
            mosol = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
            # self.transform_from_mosoli_to_chunk(mosol)
            return mosol

    def transform_from_mosoli_to_chunk(self):
        """
        Transforms the mosoli triangle to the chunk.
        """
        mosol = self.mosoli_triangle()
        polygon_xy = self.polygon().exterior.coords.xy
        polygon_x = polygon_xy[0]
        polygon_y = polygon_xy[1]
        homogenous_chunk = np.zeros((3, 3))
        homogenous_chunk[:, 0] = polygon_x[:-1]
        homogenous_chunk[:, 1] = polygon_y[:-1]
        homogenous_chunk[:, 2] = 1
        self.transformation = homogenous_chunk @ np.linalg.inv(mosol)
        return self.transformation
        
    def line_angle_calc(self):
        horizon_line = [1, 0]
        om_dir = np.array(self.om)[1]-np.array(self.om)[0]
        oc_dir = np.array(self.oc)[1]-np.array(self.oc)[0]
        dot_product = np.dot(om_dir, oc_dir)
        om_norm = np.linalg.norm(om_dir)
        oc_norm = np.linalg.norm(oc_dir)
        angle = np.arccos(dot_product/(om_norm*oc_norm))
        return angle
    
    def tri_type(self):
        """
        Returns the type of the triangle.
        """
        if self.line_angle_calc()>np.pi/4:
            return 0
        else:
            return 1
    
    def chunk_AR(self):
        """
        Returns the aspect ratio of the chunk.
        """
        if self.tri_type():
            return (1/np.tan(self.line_angle_calc()))
        else:
            return np.tan(self.line_angle_calc())
    
    def polygon(self):
        """
        Returns the polygon.
        """
        return Polygon([self.corner, self.centroid, self.projection])
    
    def inference_data(self, Re, kappa, x_limit, y_limit, grid_size):
        """
        Returns data readu to be used in inference
        """
        AR = self.chunk_AR()
        input_data = np.zeros((grid_size**2, 5))
        input_data[:, 0] = AR
        input_data[:, 1] = Re
        input_data[:, 2] = kappa
        x_range = np.linspace(0.1, x_limit, grid_size)
        y_range = np.linspace(0.1, y_limit, grid_size)
        mesh_grid = np.array([np.meshgrid(x_range, y_range)])
        mesh_grid = mesh_grid.reshape(2, grid_size**2).T
        self.mesh_grid = mesh_grid
        input_data[:, 3] = mesh_grid[:, 0]
        input_data[:, 4] = mesh_grid[:, 1]
        return input_data

    def chunk_lift(self):
        """
        Returns the lift of the chunk.
        """
        model_address = '/Users/venus/AI_lift/multi_section/model/model_state_dict_3Apr_mm'
        model = inference.Lift_base_network()
        model.load_state_dict(torch.load(model_address))
        input_params = np.loadtxt(os.path.join(CONF.DATA_DIR, "input_params/input1.csv"), delimiter=',')
        Re = input_params[0]
        kappa = input_params[1]
        x_limit = input_params[2]
        y_limit = input_params[3]
        grid_size = int(input_params[4])
        Data = self.inference_data(Re, kappa, x_limit, y_limit, grid_size)
        self.lower_tri_mask, self.upper_tri_mask = utils.mask_section(Data[:, -2:])
        Data = inference.preprocess(Data)
        Cl_map = inference.inference(Data, model)
        self.Cl_lower = Cl_map[self.lower_tri_mask]
        self.Cl_upper = Cl_map[self.upper_tri_mask]
        if self.tri_type():
            self.plot_quiver(self.Cl_lower, self.lower_tri_mask)
            return self.Cl_lower, self.lower_tri_mask
        else:
            self.plot_quiver(self.Cl_upper, self.upper_tri_mask)
            return self.Cl_upper, self.upper_tri_mask
    
    def plot_quiver(self, lift, mask):
        """
        Plots the quiver plot.
        """

        plt.quiver([self.mesh_grid[:, 0][mask], self.mesh_grid[:, 1][mask]], lift[:, 0], lift[:, 1])
