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
            mosol = np.array([[self.AR, 1, 1], [0, 0, 1], [self.AR, 0 ,1], [self.AR, 1, 1]])
        else:
            # mosol = np.array([[self.x_limit, self.y_limit, 1], [0, 0, 1], [0, self.y_limit ,1], [self.x_limit, self.y_limit, 1]])
            mosol = np.array([[self.AR, 1, 1], [0, 0, 1], [0, 1, 1], [self.AR, 1, 1]])
        # self.transform_from_mosoli_to_chunk(mosol)
        return mosol.T

    def transform_from_mosoli_to_chunk(self):
        """
        Transforms the mosoli triangle to the chunk.
        """
        mosol = self.mosoli_triangle()
        # plt.scatter(mosol[0, :], mosol[1, :], color = "green")
        polygon_xy = self.polygon().exterior.coords.xy
        polygon_x = polygon_xy[0]
        polygon_y = polygon_xy[1]
        homogenous_coord = np.ones(4)
        homogenous_chunk = np.stack((polygon_x, polygon_y, homogenous_coord), axis=0)
        # plt.scatter(homogenous_chunk[0, :], homogenous_chunk[1, :], color='y')
        self.transformation = homogenous_chunk[:, :-1] @ np.linalg.inv(mosol[:, :-1])
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
        self.AR = AR
        input_data[:, 1] = Re
        input_data[:, 2] = kappa
        self.kappa = kappa
        x_range = np.linspace(0.001, AR-kappa-0.1, grid_size)
        y_range = np.linspace(0.001, 0.9-kappa, grid_size)
        # x_range = np.linspace(0.001, self.x_limit-kappa, grid_size)
        # y_range = np.linspace(0.001, self.y_limit-kappa, grid_size)
        mesh_grid = np.array([np.meshgrid(x_range, y_range)])
        mesh_grid = mesh_grid.reshape(2, grid_size**2).T
        self.mesh_grid = mesh_grid

        input_data[:, 3] = mesh_grid[:, 1]
        input_data[:, 4] = mesh_grid[:, 0]
        return input_data

    def chunk_lift(self, id):
        """
        Returns the lift of the chunk.
        """
        self.id = id
        model_address = '/Users/venus/AI_lift/multi_section/model/model_state_dict_3Apr_mm'
        model = inference.Lift_base_network()
        model.load_state_dict(torch.load(model_address))
        input_params = np.loadtxt(os.path.join(CONF.DATA_DIR, "input_params/input1.csv"), delimiter=',')
        Re = input_params[0]
        kappa = input_params[1]
        if self.tri_type():
            self.x_limit = self.om.length
            self.y_limit = np.sqrt(self.oc.length**2-self.om.length**2)
        else:
            self.x_limit = np.sqrt(self.oc.length**2-self.om.length**2)
            self.y_limit = self.om.length 
        grid_size = int(input_params[4])
        Data = self.inference_data(Re, kappa, self.x_limit, self.y_limit, grid_size)
        self.lower_tri_mask, self.upper_tri_mask = utils.mask_section(Data[:, -2:], [self.x_limit, self.y_limit])
        # plt.scatter(self.mesh_grid[:, 0], self.mesh_grid[:, 1], color="k")
        # plt.scatter(self.mesh_grid[self.lower_tri_mask, 0], self.mesh_grid[self.lower_tri_mask, 1], color="r")
        self.base_triangle(Data)
        Data = inference.preprocess(Data)
        Cl_map = inference.inference(Data, model)
        # Cl_map = np.loadtxt("/Users/venus/AI_lift/evaluation/inference_data/Cl_map_test_infer.csv")
        self.Cl_lower = Cl_map[self.lower_tri_mask]
        self.Cl_upper = Cl_map[self.upper_tri_mask]
        if self.tri_type():
            chunk_CL =  self.plot_quiver(self.Cl_lower, self.lower_tri_mask)
            return chunk_CL
        else:
            chunk_CL = self.plot_quiver(self.Cl_upper, self.upper_tri_mask)
            return chunk_CL
    
    def base_triangle(self, Data):
        if self.tri_type():
            self.mask = self.lower_tri_mask
            self.base_points = self.mesh_grid[self.mask]
        else:
            self.mask = self.upper_tri_mask
            self.base_points = self.mesh_grid[self.mask]
        # plt.scatter(self.base_points[:, 0], self.base_points[:, 1], s=1, c='k')

    def plot_quiver(self, lift, mask):
        """
        Plots the quiver plot.
        """
        
        # plt.scatter(self.mesh_grid[mask, 0], self.mesh_grid[mask, 1])
        self.transform_from_mosoli_to_chunk()
        transformed_mesh_grid = self.transformation @ np.array([self.mesh_grid[mask, 0], self.mesh_grid[mask, 1], np.ones(self.mesh_grid[mask,:].shape[0])])
        transformed_mesh_grid[:2, :] = transformed_mesh_grid[:2, :]/transformed_mesh_grid[2, :]
        transformed_lift = self.transformation @ np.c_[lift[:, 1],lift[:, 0], np.ones(lift.shape[0])].T
        transformed_lift[:2, :] = transformed_lift[:2, :]/transformed_lift[2, :]
        # plt.scatter(transformed_mesh_grid.T[:, 0], transformed_mesh_grid.T[:, 1], color = 'red')
        # X, Y = np.meshgrid(transformed_mesh_grid.T[:, 0], transformed_mesh_grid.T[:, 1])
        x = transformed_mesh_grid.T[:, 0]
        y = transformed_mesh_grid.T[:, 1]
        u = transformed_lift.T[:, 0]
        v = transformed_lift.T[:, 1]

        # U, V = np.meshgrid(transformed_lift[0, :], transformed_lift[1, :])
        # plt.quiver(x, y, u, v, scale=20, headwidth = 3, width = 0.005, color='blue')
        plt.quiver(x, y, u, v)
        print("Plotting quiver plot")
        chunk_CL = np.stack((x, y, u, v), axis = 1)
        # np.savetxt(os.path.join(CONF.OUTPUT_DIR, f"chunk_{self.id}_CL.csv"), chunk_CL, delimiter=',')
        return chunk_CL
    