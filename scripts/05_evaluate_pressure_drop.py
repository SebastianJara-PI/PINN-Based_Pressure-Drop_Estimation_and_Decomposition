import csv
import sys
import time
from datetime import datetime
from pathlib import Path

# Resolve the repository and load helper modules from src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from NN import MLP
from utils import predict_pressure_on_centerline, predict_pressure_on_tubular_region
from visualization_plots import plot_pressure_drop_panels

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################
# Case 

CASE_NAME = "cfd_9mm"
PRESSURE_RUN_NAME = "seed_001"
PLOT_TIME = 0.15
SPATIAL_PANEL_WIDTH = 2.0
DROP_PANEL_WIDTH = 1.0
PRESSURE_RUN_DIR = (
    PROJECT_ROOT / "outputs" / CASE_NAME / "pressure" / PRESSURE_RUN_NAME
)
##########################################################
# Paths
data_dir = PROJECT_ROOT / "data" / CASE_NAME / "processed"
segmentation_path = PROJECT_ROOT / "data" / CASE_NAME / "raw" / f"{CASE_NAME}.npy"
centerline_path = data_dir / "centerline_xyz.npy"
data_path = data_dir / "tubular_region_0.002_cfd_9mm.npy" 
pressure_ckpt_path = PRESSURE_RUN_DIR / "checkpoints" / "best_pressure.pth"
path_dic = PRESSURE_RUN_DIR / "sampling" / "dic_adim_norm.npz"



print(f"Centerline path: {centerline_path}")
print(f"Data path: {data_path}")
print(f"Pressure checkpoint path: {pressure_ckpt_path}")


##########################################################
# Geometry P1-P2 and tangents for the non-clinical score

centerline_xyz = np.load(centerline_path, allow_pickle=False)
data_grid = np.load(data_path, allow_pickle=False)
segmentation_grid = np.load(segmentation_path, mmap_mode="r", allow_pickle=False)

print(f"Centerline XYZ shape: {centerline_xyz.shape}")
print(f"Data grid shape: {data_grid.shape}")

##########################################################
# Load mode
ckpt = torch.load(pressure_ckpt_path, map_location="cpu")
model_state_dict = ckpt["model_state_dict"]

### net 
net_pres = MLP(
    input_size=4,
    output_size=3,
    hidden_layers=3,
    hidden_units=32,
    activation_fn=nn.SiLU(),
)

net_pres.load_state_dict(model_state_dict)

# Adimensional factors
dic_adim = np.load(path_dic, allow_pickle=True)

factor_names = (
    "U", "L", "T", "rho", "mu", "p0", "Re",
    "mu_x", "mu_y", "mu_z", "mu_t",
    "sigma_x", "sigma_y", "sigma_z", "sigma_t",
)
(
    U, L, T, rho, _ , p0, Re,
    mu_x, mu_y, mu_z, mu_t,
    sigma_x, sigma_y, sigma_z, sigma_t,
) = (
    float(np.asarray(dic_adim[name]).squeeze())
    for name in factor_names
)

mmhg_scale = p0 / 133.322

# print(f"U: {U}, L: {L}, T: {T}, rho: {rho}, mu: {mu}, p0: {p0}, Re: {Re}")
# print(f"mu_x: {mu_x}, mu_y: {mu_y}, mu_z: {mu_z}, mu_t: {mu_t}")
# print(f"sigma_x: {sigma_x}, sigma_y: {sigma_y}, sigma_z: {sigma_z}, sigma_t: {sigma_t}")
    

##########################################################
# Prediction total pressure and components on centerline 

Data_pred_pres_center = predict_pressure_on_centerline(
    data_grid=data_grid,
    centerline_xyz=centerline_xyz,
    net_pres=net_pres,
    L=L,
    T=T,
    mu=[mu_x, mu_y, mu_z, mu_t],
    sigma=[sigma_x, sigma_y, sigma_z, sigma_t],
    mmhg_scale=mmhg_scale,
)


Data_pred_pres = predict_pressure_on_tubular_region(
    data_grid=data_grid,
    net_pres=net_pres,
    L=L,
    T=T,
    mu=[mu_x, mu_y, mu_z, mu_t],
    sigma=[sigma_x, sigma_y, sigma_z, sigma_t],
    mmhg_scale=mmhg_scale,
)


#print(P_components_center.shape)
#print(X_norm)
print(Data_pred_pres_center.shape)


##########################################################
# Plots

# Select points on centerline
idx_p1 = 35
# idx_p2 = 170
idx_p2 = 70

# P1 point
P1_manual = centerline_xyz[idx_p1,:3]
P1 = P1_manual.reshape(1, 3)

pressure_p1_acc = Data_pred_pres_center[:, idx_p1, 4]
pressure_p1_adv = Data_pred_pres_center[:, idx_p1, 5]
pressure_p1_vis = Data_pred_pres_center[:, idx_p1, 6]
pressure_p1_total = Data_pred_pres_center[:, idx_p1, 7]

# P2 point
P2_manual = centerline_xyz[idx_p2,:3]
P2 = P2_manual.reshape(1, 3)

pressure_p2_acc = Data_pred_pres_center[:, idx_p2, 4]
pressure_p2_adv = Data_pred_pres_center[:, idx_p2, 5]
pressure_p2_vis = Data_pred_pres_center[:, idx_p2, 6]
pressure_p2_total = Data_pred_pres_center[:, idx_p2, 7]

# Pressure drops between P1 and P2
drop_acc = pressure_p1_acc - pressure_p2_acc
drop_adv = pressure_p1_adv - pressure_p2_adv
drop_vis = pressure_p1_vis - pressure_p2_vis
drop_total = pressure_p1_total - pressure_p2_total

# Presiones CFD 
xyz_ref = data_grid[0, :, 0:3]
time_p = data_grid[:, 0, 3]

dist2_p1 = np.sum((xyz_ref - P1)**2, axis=1)
dist2_p2 = np.sum((xyz_ref - P2)**2, axis=1)

idx_cfd_p1 = int(np.argmin(dist2_p1))
idx_cfd_p2 = int(np.argmin(dist2_p2))

P1_cfd = xyz_ref[idx_cfd_p1,:]
P2_cfd = xyz_ref[idx_cfd_p2,:]

p1_cfd = data_grid[:, idx_cfd_p1, -1]
p2_cfd = data_grid[:, idx_cfd_p2, -1]
drop_cfd = p1_cfd - p2_cfd

# set plotting parameters
plot_time_idx = int(np.argmin(np.abs(time_p - PLOT_TIME)))
plot_time = float(time_p[plot_time_idx])
spatial_points = Data_pred_pres[plot_time_idx, :, :3]
predicted_total_pressure = Data_pred_pres[plot_time_idx, :, 7]
segmentation_xyz = np.asarray(segmentation_grid[0, :, :3])
figures_dir = PROJECT_ROOT / "outputs" / CASE_NAME / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
figure_path = figures_dir / (
    f"F_pressure_drop_{PRESSURE_RUN_NAME}_p1_{idx_p1}_p2_{idx_p2}_"
    f"t_{plot_time:.4f}.png"
)

fig, axes = plot_pressure_drop_panels(
    segmentation_xyz=segmentation_xyz,
    spatial_points=spatial_points,
    predicted_total_pressure=predicted_total_pressure,
    centerline_xyz=centerline_xyz,
    p1_xyz=P1_manual,
    p2_xyz=P2_manual,
    time_value=plot_time,
    time_values=time_p,
    drop_components={
        "acc": drop_acc,
        "adv": drop_adv,
        "vis": drop_vis,
    },
    drop_total=drop_total,
    drop_cfd=drop_cfd,
    p1_index=idx_p1,
    p2_index=idx_p2,
    cfd_p1_index=idx_cfd_p1,
    cfd_p2_index=idx_cfd_p2,
    figure_size=(20, 7),
    panel_widths=(SPATIAL_PANEL_WIDTH, DROP_PANEL_WIDTH),
    view_angle=(45, 45),
    max_segmentation_points=50000,
    output_path=figure_path,
)
plt.show()

