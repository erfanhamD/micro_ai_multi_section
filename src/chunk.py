import numpy as np
from shapely.geometry import LineString, Polygon
class Chunk:
    def __init__(self, centroid, corner, projection):
        self.projection = projection
        self.centroid = centroid
        self.corner = corner
        self.oc = LineString([self.corner, self.centroid])
        self.om = LineString([self.projection, self.centroid])

    
    def line_angle_calc(self, line):
        horizon_line = [1, 0]
        line_origin = np.array(line)[1]-np.array(line)[0]
        dot_product = np.dot(horizon_line, line_origin)
        line_norm = np.linalg.norm(np.array(line)[1]-np.array(line)[0])
        angle = np.arccos(dot_product/line_norm)
        return angle
    
    def tri_type(self):
        """
        Returns the type of the triangle.
        """
        if self.line_angle_calc(self.om)>self.line_angle_calc(self.oc):
            return 1
        else:
            return 0
    
    def chunk_AR(self):
        """
        Returns the aspect ratio of the chunk.
        """
        x_dist = self.projection.distance(self.corner)
        y_dist = self.centroid.distance(self.projection)
        if self.tri_type():
            return x_dist/y_dist
        else:
            return y_dist/x_dist
    
    def polygon(self):
        """
        Returns the polygon.
        """
        return Polygon([self.corner, self.centroid, self.projection])
        