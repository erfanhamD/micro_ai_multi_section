import numpy as np
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import utils
from chunk import Chunk
from figures import BLUE, RED, SIZE, set_limits, plot_coords, color_isvalid
from descartes.patch import PolygonPatch

corners = utils.load_section_geometry("section_3.csv")
polygon = Polygon(corners)
vertices = polygon.boundary.coords
centeroid = polygon.centroid
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
print("lsfajlfjalfj")