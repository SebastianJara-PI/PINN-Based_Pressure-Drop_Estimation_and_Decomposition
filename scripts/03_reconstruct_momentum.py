import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from NN import MLP


torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################
# Case and experiment configuration

# Case configuration
CASE_NAME = "cfd_9mm"
# CASE_NAME = "cfd_11mm"
# CASE_NAME = "cfd_13mm"
# CASE_NAME = "cfd_normal"
VELOCITY_RUN_NAME = "seed_001"
DATA_FILENAME = f"tubular_region_0.002_{CASE_NAME}.npy"
training_seed = 1
BATCH_SIZE = 1500  # reduce if memory is insufficient for second derivatives

# Paths
run_dir = PROJECT_ROOT / "outputs" / CASE_NAME / "velocity" / VELOCITY_RUN_NAME
best_ckpt_path = run_dir / "checkpoints" / "best_velocity.pth"
normalization_path = run_dir / "sampling" / "dic_adim_norm.npz"
data_path = PROJECT_ROOT / "data" / CASE_NAME / "processed" / DATA_FILENAME
derived_dir = (
    PROJECT_ROOT / "outputs" / CASE_NAME / "momentum_terms" / VELOCITY_RUN_NAME
)

for path in (best_ckpt_path, normalization_path, data_path):
    if not path.is_file():
        raise FileNotFoundError(f"Does not exist: {path}")

# Autograd function for computing gradients
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


##########################################################
# Load checkpoint and reconstruct network

checkpoint = torch.load(best_ckpt_path, map_location="cpu")
extra_state = checkpoint["extra_state"]
if int(extra_state["training_seed"]) != training_seed:
    raise ValueError("The checkpoint training seed does not match training_seed.")

dic_adim_norm = extra_state["dic_adim_norm"]

mu_x = float(dic_adim_norm["mu_x"])
mu_y = float(dic_adim_norm["mu_y"])
mu_z = float(dic_adim_norm["mu_z"])
mu_t = float(dic_adim_norm["mu_t"])

sigma_x = float(dic_adim_norm["sigma_x"])
sigma_y = float(dic_adim_norm["sigma_y"])
sigma_z = float(dic_adim_norm["sigma_z"])
sigma_t = float(dic_adim_norm["sigma_t"])

L = float(extra_state["L"])
T = float(extra_state["T"])
U = float(extra_state["U"])
rho = float(extra_state["rho"])
mu = float(extra_state["mu"])
Re = float(extra_state["Re"])

parameters = {
    key: float(dic_adim_norm[key])
    for key in (
        "mu_x", "mu_y", "mu_z", "mu_t",
        "sigma_x", "sigma_y", "sigma_z", "sigma_t",
    )
}
nu = mu / rho

net = MLP(
    input_size=4,
    output_size=3,
    hidden_layers=4,
    hidden_units=64,
    activation_fn=nn.SiLU(),
).to(device)

net.load_state_dict(checkpoint["model_state_dict"])
net.eval()

print("=" * 90)
print(f"Device:                  {device}")
print(f"Run directory:           {run_dir}")
print(f"Best checkpoint:         {best_ckpt_path}")
print(f"Best epoch:              {checkpoint['epoch']}")
print(f"Best validation metric:  {checkpoint['validation_metric']:.6e}")
print(f"L={L}, T={T}, U={U}, Re={Re:.3f}, nu={nu:.6e}")
print("=" * 90)


########################################
### Load data
Data = np.load(data_path, allow_pickle=False)
Nt, Np, ncols = Data.shape


Data_flat = Data.reshape(-1, ncols)

coords_xyz_t = Data_flat[:, 0:4]  # physical coordinates: x,y,z,t

# Normalization of inputs
x_phys = coords_xyz_t[:, 0:1]
y_phys = coords_xyz_t[:, 1:2]
z_phys = coords_xyz_t[:, 2:3]
t_phys = coords_xyz_t[:, 3:4]

x_hat = (x_phys / L - mu_x) / sigma_x
y_hat = (y_phys / L - mu_y) / sigma_y
z_hat = (z_phys / L - mu_z) / sigma_z
t_hat = (t_phys / T - mu_t) / sigma_t

X_input = np.concatenate([x_hat, y_hat, z_hat, t_hat], axis=1)

N_total = X_input.shape[0]

print(f"Data shape:              {Data.shape}")
print(f"Flattened points:        {N_total}")
print(f"Batch size:              {BATCH_SIZE}")


# Output buffers
acc_prime_all = np.zeros((N_total, 3), dtype=np.float64)
adv_prime_all = np.zeros((N_total, 3), dtype=np.float64)
vis_prime_all = np.zeros((N_total, 3), dtype=np.float64)

# Loop over batches
for start in range(0, N_total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N_total)

    Xb = X_input[start:end]

    xb = torch.tensor(Xb[:, 0:1], dtype=torch.float64, device=device, requires_grad=True)
    yb = torch.tensor(Xb[:, 1:2], dtype=torch.float64, device=device, requires_grad=True)
    zb = torch.tensor(Xb[:, 2:3], dtype=torch.float64, device=device, requires_grad=True)
    tb = torch.tensor(Xb[:, 3:4], dtype=torch.float64, device=device, requires_grad=True)

    # NN: output u', v', w'
    uvw_prime = net(xb, yb, zb, tb)
    u_p = uvw_prime[:, 0:1]
    v_p = uvw_prime[:, 1:2]
    w_p = uvw_prime[:, 2:3]

    # Accelerative term: ∂u'/∂t' 
    u_t_hat = grad(u_p, tb)
    v_t_hat = grad(v_p, tb)
    w_t_hat = grad(w_p, tb)
    u_t_prime = u_t_hat / sigma_t
    v_t_prime = v_t_hat / sigma_t
    w_t_prime = w_t_hat / sigma_t
    acc_prime = torch.cat([u_t_prime, v_t_prime, w_t_prime], dim=1)

    # First-order spatial derivatives: ∂u'/∂x', etc.
    u_x_hat = grad(u_p, xb)
    u_y_hat = grad(u_p, yb)
    u_z_hat = grad(u_p, zb)
    v_x_hat = grad(v_p, xb)
    v_y_hat = grad(v_p, yb)
    v_z_hat = grad(v_p, zb)
    w_x_hat = grad(w_p, xb)
    w_y_hat = grad(w_p, yb)
    w_z_hat = grad(w_p, zb)

    u_x_prime = u_x_hat / sigma_x
    u_y_prime = u_y_hat / sigma_y
    u_z_prime = u_z_hat / sigma_z
    v_x_prime = v_x_hat / sigma_x
    v_y_prime = v_y_hat / sigma_y
    v_z_prime = v_z_hat / sigma_z
    w_x_prime = w_x_hat / sigma_x
    w_y_prime = w_y_hat / sigma_y
    w_z_prime = w_z_hat / sigma_z

    # Advective term: (u' · ∇')u'
    adv_u = u_p * u_x_prime + v_p * u_y_prime + w_p * u_z_prime
    adv_v = u_p * v_x_prime + v_p * v_y_prime + w_p * v_z_prime
    adv_w = u_p * w_x_prime + v_p * w_y_prime + w_p * w_z_prime

    adv_prime = torch.cat([adv_u, adv_v, adv_w], dim=1)

    # Laplacian: ∇'^2 u'
    u_xx_prime = grad(u_x_hat, xb) / (sigma_x ** 2)
    u_yy_prime = grad(u_y_hat, yb) / (sigma_y ** 2)
    u_zz_prime = grad(u_z_hat, zb) / (sigma_z ** 2)

    v_xx_prime = grad(v_x_hat, xb) / (sigma_x ** 2)
    v_yy_prime = grad(v_y_hat, yb) / (sigma_y ** 2)
    v_zz_prime = grad(v_z_hat, zb) / (sigma_z ** 2)

    w_xx_prime = grad(w_x_hat, xb) / (sigma_x ** 2)
    w_yy_prime = grad(w_y_hat, yb) / (sigma_y ** 2)
    w_zz_prime = grad(w_z_hat, zb) / (sigma_z ** 2)

    lap_u_prime = u_xx_prime + u_yy_prime + u_zz_prime
    lap_v_prime = v_xx_prime + v_yy_prime + v_zz_prime
    lap_w_prime = w_xx_prime + w_yy_prime + w_zz_prime

    vis_prime = (1.0 / Re) * torch.cat(
        [lap_u_prime, lap_v_prime, lap_w_prime],
        dim=1,
    )

    acc_prime_all[start:end] = acc_prime.detach().cpu().numpy()
    adv_prime_all[start:end] = adv_prime.detach().cpu().numpy()
    vis_prime_all[start:end] = vis_prime.detach().cpu().numpy()

    if start % (10 * BATCH_SIZE) == 0 or end == N_total:
        print(f"Processed {end}/{N_total} points")

# Scale to MKS
scale_mks = (U ** 2) / L
acc_mks = scale_mks * acc_prime_all
adv_mks = scale_mks * adv_prime_all
vis_mks = scale_mks * vis_prime_all


# Reassemble arrays [Nt, Np, 7]
# columns = x,y,z,t, term_x, term_y, term_z
Data_acc = np.concatenate([coords_xyz_t, acc_prime_all], axis=1).reshape(Nt, Np, 7)
Data_adv = np.concatenate([coords_xyz_t, adv_prime_all], axis=1).reshape(Nt, Np, 7)
Data_visc = np.concatenate([coords_xyz_t, vis_prime_all], axis=1).reshape(Nt, Np, 7)
Data_acc_mks = np.concatenate([coords_xyz_t, acc_mks], axis=1).reshape(Nt, Np, 7)
Data_adv_mks = np.concatenate([coords_xyz_t, adv_mks], axis=1).reshape(Nt, Np, 7)
Data_visc_mks = np.concatenate([coords_xyz_t, vis_mks], axis=1).reshape(Nt, Np, 7)


# Save the six arrays in the folder of the selected case and training.
derived_dir.mkdir(parents=True, exist_ok=True)
outputs = {
    "Data_acc.npy": Data_acc,
    "Data_adv.npy": Data_adv,
    "Data_visc.npy": Data_visc,
    "Data_acc_mks.npy": Data_acc_mks,
    "Data_adv_mks.npy": Data_adv_mks,
    "Data_visc_mks.npy": Data_visc_mks,
}
for filename, array in outputs.items():
    np.save(derived_dir / filename, array)

metadata = {
    "case": CASE_NAME,
    "training_seed": training_seed,
    "velocity_run": VELOCITY_RUN_NAME,
    "checkpoint_path": str(best_ckpt_path),
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "validation_metric": float(checkpoint["validation_metric"]),
    "normalization_path": str(normalization_path),
    "data_path": str(data_path),
    "batch_size": BATCH_SIZE,
    "parameters": dic_adim_norm,
    "output_shape": [Nt, Np, 7],
    "columns": ["x", "y", "z", "t", "term_x", "term_y", "term_z"],
    "coordinate_units": ["m", "m", "m", "s"],
    "term_units": {"without_mks": "dimensionless", "mks": "m/s^2"},
    "mks_multiplier": scale_mks,
    "terms": {
        "acc": "du_prime/dt_prime",
        "adv": "(u_prime dot grad_prime) u_prime",
        "visc": "(1/Re) laplacian_prime(u_prime)",
    },
    "files": list(outputs),
}
with (derived_dir / "reconstruction_metadata.json").open("w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print(f"\nResults saved in: {derived_dir}")


def print_stats(name, array):
    values = array[:, :, 4:7]
    print(
        f"{name}: shape={array.shape}, "
        f"NaN={np.isnan(values).sum()}, Inf={np.isinf(values).sum()}, "
        f"min={np.nanmin(values):.6e}, max={np.nanmax(values):.6e}"
    )

print_stats("Data_acc_mks", Data_acc_mks)
print_stats("Data_adv_mks", Data_adv_mks)
print_stats("Data_visc_mks", Data_visc_mks)
print_stats("Data_acc", Data_acc)
print_stats("Data_adv", Data_adv)
print_stats("Data_visc", Data_visc)















