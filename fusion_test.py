import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import csv

data = np.genfromtxt("city_1_0/velo_lidar/000679.csv",delimiter=",")
points = data[:,0:3]

import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([pcd])

#voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.40)
#o3d.visualization.draw_geometries([voxel_grid])





