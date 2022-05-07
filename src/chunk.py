import numpy as np
from shapely.geometry import LineString, Polygon

from utils import line_angle_calc
class Chunk:
    def __init__(self, centroid, corner, projection):
        self.projection = projection
        self.centroid = centroid
        self.corner = corner
        self.oc = LineString([self.corner, self.centroid])
        self.om = LineString([self.projection, self.centroid])

    
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
        input_data[:, 3] = mesh_grid[:, 0]
        input_data[:, 4] = mesh_grid[:, 1]
        return input_data
