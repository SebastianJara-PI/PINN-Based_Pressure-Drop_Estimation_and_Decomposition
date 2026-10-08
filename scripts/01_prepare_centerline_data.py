from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

# Folder root
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Import visualization functions from src
sys.path.insert(0, str(PROJECT_DIR / "src"))
from visualization_plots import plot_velocity_panels, plot_pressure_panels, plot_segmentation_with_centerline, plot_tubular_region, plot_centerline_velocity_panels
from utils import build_centerline_from_pointcloud

# Chosen Case
CASE_NAME = "cfd_9mm"
# CASE_NAME = "cfd_11mm"
# CASE_NAME = "cfd_13mm"
# CASE_NAME = "cfd_normal"

# Paths
CASE_DIR = PROJECT_DIR / "data" / CASE_NAME
DATA_PATH = CASE_DIR / "raw" / f"{CASE_NAME}.npy"
FIGURE_PATH = PROJECT_DIR / "outputs" / CASE_NAME / "figures"
FIGURE_PATH.mkdir(parents=True, exist_ok=True)

# Load data
Data_grid = np.load(DATA_PATH)
Nt = Data_grid.shape[0]
Np = Data_grid.shape[1]
Nf = Data_grid.shape[2]

print("Loaded file:", DATA_PATH)
print(
    f"Dimensions: ({Nt}, {Np}, {Nf}) — {Nt} time steps\n"
    f"{Np} spatial points per time step, and {Nf} variables per point\n"
    "Columns 1-3: spatial coordinates (x, y, z)\n"
    "column 4: time (t)\n"
    "columns 5-7: velocity components (u, v, w)\n"
    "column 8: pressure (p).\n"
)

##########################################################
# Data visualization and centerline preparation

# Plot vector velocity raw 
fig, axes = plot_velocity_panels(
    Data_grid,
    time_index=7,
    view_angles=((45, 45), (20, 130)),
    vector_scale=0.01,
    normalize_vectors=False,
    max_vectors=5000,
    max_segmentation_points=50000,
    output_path= FIGURE_PATH / "A_velocity_vectors_raw.png",
)

plt.show()

# Plot pressure field CFD
fig, axes = plot_pressure_panels(
    Data_grid,
    time_index=10,
    view_angles=((45, 45), (20, 135)),
    max_points=50000,
    point_size=3,
    cmap_name="viridis",
    pressure_unit="mmHg",  # Si tus datos están en pascales.
    output_path=FIGURE_PATH / "B_pressure_panels.png",
)

plt.show()


##########################################################
# Centerline Builder
centerline_xyz, skel_xyz, mask, origin = build_centerline_from_pointcloud(
    Data_grid,
    voxel_size=0.001,  
    padding=2,
    smooth=0.0001
)
print("Number of points in the centerline:", len(centerline_xyz))
print("Centerline shape:", centerline_xyz.shape)

# Plot segmentation with centerline
fig, ax = plot_segmentation_with_centerline(
    Data_grid,
    centerline_xyz,
    skel_xyz=skel_xyz,
    view_angle=(110, 90),         
    max_segmentation_points=50000,
    show_skeleton=False,
    output_path=FIGURE_PATH / "C_segmentation_with_centerline.png",            
)

plt.show()

##########################################################
# Tubular Region Builder
tree = cKDTree(Data_grid[0, :, :3])
radius =  0.002     # 2 mm  radius of the tubular region

idx_near_list = tree.query_ball_point(centerline_xyz, r=radius)
idx_near_unique = np.unique( np.concatenate([idx for idx in idx_near_list if len(idx) > 0]) )

coords_tube = Data_grid[0, idx_near_unique, :3]

# Order nearby points according to their position along the centerline
tree_centerline = cKDTree(centerline_xyz)
dist_to_cl, idx_cl_nearest = tree_centerline.query(coords_tube)
order = np.argsort(idx_cl_nearest)

coords_tube_ordered = coords_tube[order]
idx_tube_ordered = idx_near_unique[order]
idx_cl_tube_ordered = idx_cl_nearest[order]

# Plot tubular region
fig, ax = plot_tubular_region(
    coords_tube_ordered,
    Data_grid,
    radius,
    view_angle=(30, 25),
    output_path=FIGURE_PATH / "D_tubular_region.png",
)

# Data corresponding to the tubular region along the centerline
data_centerline = Data_grid[:, idx_tube_ordered, :]
print(data_centerline.shape)

##########################################################
# Centerline Velocity Panels and centerline

# Plot centerline velocity panels
fig, ax =  plot_centerline_velocity_panels(
    Data_grid,
    data_centerline,
    time_index=7,
    view_angles=((45, 45), (20, 135)),
    vector_scale=0.009,
    normalize_vectors=False,
    max_vectors=15000,
    max_segmentation_points=50000,
    output_path=FIGURE_PATH / "E_tubular_region_velocity_panels.png",
)

plt.show()


##########################################################
# Save Tubular Region data

# Save processed data
PROCESSED_PATH = PROJECT_DIR / "data" / CASE_NAME / "processed"
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

np.save(PROCESSED_PATH / "centerline_xyz.npy", centerline_xyz)
np.save(PROCESSED_PATH / f"tubular_region_{radius:.3f}_{CASE_NAME}.npy", data_centerline)
print(f"Saved tubular region data: {PROCESSED_PATH / f'tubular_region_{radius:.3f}_{CASE_NAME}.npy'}")