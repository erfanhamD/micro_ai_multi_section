import numpy as np
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt
import utils
from chunk import Chunk
from figures import BLUE, RED, SIZE, set_limits, plot_coords, color_isvalid
from descartes.patch import PolygonPatch



corners = utils.load_section_geometry("section_2.csv")
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
        chunks.append(chunk)
        polygon = chunk.polygon()
        plot_coords(ax, polygon.exterior)
        patch = PolygonPatch(polygon, facecolor=chunk_color_list[jdx], edgecolor=color_isvalid(polygon, valid=chunk_color_list[jdx]), alpha=0.5, zorder=2)
        ax.add_patch(patch)
# projection_lines = [LineString([projections[i], centeroid]) for i in range(len(projections))]
# angles = np.array([utils.line_angle_calc(projection_lines[i]) for i in range(len(projection_lines))])*180/np.pi
# for j in range(4):
#     oc = centeroid_corner_line[j]
#     om = projection_lines[j]
#     tri_type = utils.tri_type(Lin)
# AR = np.array([utils.compute_aspect_ratio(projections[i], centeroid, vertices[i], angles[i]<90) for i in range(len(projections))])
# print(angles)
# AR_dict = {}
# for chunc_id in range(len(edges)):
#     AR_dict[chunc_id] = utils.compute_aspect_ratio(projections[chunc_id], centeroid, edges[chunc_id])

chunks[0].chunk_AR()
# plt.figure()
plt.scatter(*polygon.exterior.xy, color = 'k')
plt.plot(corners[:,0], corners[:,1], '--k')
plt.scatter(*centeroid.xy, color  = 'r', linewidths=30, alpha = 0.3)
for i in range(len(projections)):
    plt.scatter(*projections[i].xy, color = 'orange')
    plt.plot(*centeroid_corner_line[i].xy, '--b')
# for i in range(len(projection_lines)):
#     plt.plot(*projection_lines[i].xy, '--b')
plt.gca().set_aspect('equal')

plt.show()
print("lsfajlfjalfj")