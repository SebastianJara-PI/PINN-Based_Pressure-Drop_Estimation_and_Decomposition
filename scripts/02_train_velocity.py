import csv
import sys
import time
from datetime import datetime
from pathlib import Path

# Folder root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Import functions from src
from loss_functions import velocity_loss, NS_loss
from loss_weights import update_loss_weights_velocity
from utils import normalize_dataset, sample_batch, train_test_split_numpy
from NN import MLP
from utils_train_velocity import (
    EarlyStopping,
    evaluate_velocity_validation,
    save_json,
    set_all_seeds,
)

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##########################################################
# Case and experiment configuration

# Case configuration

CASE_NAME = "cfd_9mm"
# CASE_NAME = "cfd_11mm"
# CASE_NAME = "cfd_13mm"
# CASE_NAME = "cfd_normal"

DATA_FILENAME = f"tubular_region_0.002_{CASE_NAME}.npy"

experiment_name = CASE_NAME
# seed that changes between runs
training_seed = 1
# seed fixed for the split
split_seed = 43
# seed for minibatch sampling
# derived from training_seed so that each run changes reproducibly
batch_seed = 10_000 + training_seed
max_epochs = 100_000
eval_every = 2_500
save_regular_every = 5_000
warmup_epochs = 10_000
patience = 10
min_delta_relative = 2e-3
validation_batch_size = 20_000

# Reproducibility
set_all_seeds(training_seed)

# Paths
data_dir = PROJECT_ROOT / "data" / CASE_NAME / "processed"
output_dir = PROJECT_ROOT / "outputs" / CASE_NAME / "velocity"
run_name = (
    f"seed_{training_seed:03d}"
)

base_dir = output_dir / run_name
checkpoints_dir = base_dir / "checkpoints"
sampling_dir = base_dir / "sampling"
logs_dir = base_dir / "logs"
checkpoints_dir.mkdir(parents=True, exist_ok=True)
sampling_dir.mkdir(parents=True, exist_ok=True)
logs_dir.mkdir(parents=True, exist_ok=True)
best_checkpoint_path = checkpoints_dir / "best_velocity.pth"
last_checkpoint_path = checkpoints_dir / "checkpoint_last.pth"
validation_csv_path = logs_dir / "validation_history.csv"
summary_json_path = logs_dir / "training_summary.json"

# Load data
data_path = data_dir / DATA_FILENAME
if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{data_path}"
    )
Data = np.load(data_path, allow_pickle=False)

print("=" * 90)
print(f"Device:                {device}")
print(f"Experiment:            {experiment_name}")
print(f"Training seed:         {training_seed}")
print(f"Split seed:            {split_seed}")
print(f"Data loaded from:      {data_path}")
print(f"Results saved in:      {base_dir}")
print(f"Data shape:            {Data.shape}")
print("=" * 90)

# Physical scales
U = 1.0
L = 0.09
T = L / U
rho = 1119.0
mu = 0.00483
p0 = rho * U**2
Re = rho * U * L / mu

# Normalization parameters
Nt, N_points, ncols = Data.shape
X_raw = Data.reshape(-1, ncols)
X_space_a = X_raw[:, 0:3] / L
X_time_a = X_raw[:, 3:4] / T
X_space_a_mean = X_space_a.mean(axis=0)
X_time_a_mean = X_time_a.mean(axis=0)
eps = 1e-14
X_space_a_std = np.maximum(X_space_a.std(axis=0), eps)
X_time_a_std = np.maximum(X_time_a.std(axis=0), eps)
mu_x, mu_y, mu_z = X_space_a_mean
mu_t = X_time_a_mean[0]
sigma_x, sigma_y, sigma_z = X_space_a_std
sigma_t = X_time_a_std[0]

dic_adim_norm = {
    "mu_x": float(mu_x),
    "mu_y": float(mu_y),
    "mu_z": float(mu_z),
    "mu_t": float(mu_t),
    "sigma_x": float(sigma_x),
    "sigma_y": float(sigma_y),
    "sigma_z": float(sigma_z),
    "sigma_t": float(sigma_t),
}
np.savez(
    sampling_dir / "dic_adim_norm.npz",
    **dic_adim_norm,
    L=L,
    T=T,
    U=U,
    rho=rho,
    mu=mu,
    p0=p0,
    Re=Re,
)

# Normalize dataset
X_norm = normalize_dataset(
    X_raw,
    L=L,
    T=T,
    U=U,
    mu_x=mu_x,
    mu_y=mu_y,
    mu_z=mu_z,
    mu_t=mu_t,
    sigma_x=sigma_x,
    sigma_y=sigma_y,
    sigma_z=sigma_z,
    sigma_t=sigma_t,
)

# Train / validation split
(
    X_train_data,
    X_val_data,
    idx_train_data,
    idx_val_data,
) = train_test_split_numpy(
    X_norm,
    train_ratio=0.90,
    seed=split_seed,
)
np.savez(
    sampling_dir / "sampling_index.npz",
    idx_train_data=idx_train_data,
    idx_val_data=idx_val_data,
    split_seed=split_seed,
)
print(f"Train samples:         {len(X_train_data)}")
print(f"Validation samples:    {len(X_val_data)}")

# Model
net = MLP(
    input_size=4,
    output_size=3,
    hidden_layers=4,
    hidden_units=64,
    activation_fn=nn.SiLU(),
).to(device)

# Optimizer 
optimizer = optim.Adam(
    net.parameters(),
    lr=1e-3,
)

# Adaptive loss weights
grad_weight_scheme = True
grad_weight_update_every = 100
grad_weight_start_epoch = 0
grad_weight_alpha = 0.8
lambda_min = 1e-3
lambda_max = 1e3
loss_weights = np.array(
    [1.0, 1.0, 1.0],
    dtype=np.float64,
)
lambda_div = 1.0
# Multiplier on the divergence at the end of the warmup period:
# a checkpoint is only accepted if rms_divergence <= max_divergence_threshold.
divergence_multiplier = 1.20


##########################################################
# Training velocity network

# Training settings
N_d = 1000      # Number of data samples per batch
N_c = 4000      # Number of colocation samples per batch

rng = np.random.default_rng(batch_seed)

early_stopping = EarlyStopping(
    checkpoint_path=best_checkpoint_path,
    patience=patience,
    warmup_epochs=warmup_epochs,
    min_delta_relative=min_delta_relative,
)

history = {
    "total_loss": [],
    "loss_data": [],
    "loss_data_u": [],
    "loss_data_v": [],
    "loss_data_w": [],
    "loss_div": [],
    "lambda_data_u": [],
    "lambda_data_v": [],
    "lambda_data_w": [],
    "lambda_div": [],
    "lr": [],
}
validation_history = []
max_divergence_threshold = None  # Maximum allowed divergence, set at the end of the warmup period
last_print_time = time.time()

# CSV header
validation_fields = [
    "epoch",
    "rmse_speed",
    "nrmse_range_speed",
    "relative_l2_speed",
    "relative_l2_velocity_vector",
    "rms_divergence",
    "mse_u",
    "mse_v",
    "mse_w",
    "divergence_constraint_satisfied",
    "max_divergence_threshold",
    "selection_score",
    "improved",
    "best_epoch",
    "best_metric",
    "evaluations_without_improvement",
]
with validation_csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=validation_fields)
    writer.writeheader()

# Training loop
for epoch in range(max_epochs + 1):
    X_data_batch, _ = sample_batch(
        X_train_data,
        N_d,
        rng,
    )
    X_colocation_batch, _ = sample_batch(
        X_train_data,
        N_c,
        rng,
    )
    X_data_batch_tensor = torch.tensor(
        X_data_batch,
        dtype=torch.float64,
        device=device,
    ).requires_grad_(True)
    X_colocation_batch_tensor = torch.tensor(
        X_colocation_batch,
        dtype=torch.float64,
        device=device,
    ).requires_grad_(True)

    # Adaptive loss weights
    if (
        grad_weight_scheme
        and epoch >= grad_weight_start_epoch
        and epoch % grad_weight_update_every == 0
    ):
        loss_weights, grad_norms = update_loss_weights_velocity(
            model=net,
            loss_weights=loss_weights,
            X_data_batch_tensor=X_data_batch_tensor,
            alpha=grad_weight_alpha,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )
    lambda_data_u = float(loss_weights[0])
    lambda_data_v = float(loss_weights[1])
    lambda_data_w = float(loss_weights[2])

    # Forward + backward
    optimizer.zero_grad(set_to_none=True)
    X_data_batch_tensor = (
        X_data_batch_tensor
        .detach()
        .clone()
        .requires_grad_(True)
    )
    X_colocation_batch_tensor = (
        X_colocation_batch_tensor
        .detach()
        .clone()
        .requires_grad_(True)
    )
    loss_u, loss_v, loss_w = velocity_loss(
        net,
        X_data_batch_tensor,
    )
    loss_data_value = loss_u + loss_v + loss_w
    loss_div = NS_loss(
        net,
        X_colocation_batch_tensor,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        sigma_z=sigma_z
    )
    total_loss = (
        lambda_data_u * loss_u
        + lambda_data_v * loss_v
        + lambda_data_w * loss_w
        + lambda_div * loss_div
    )
    total_loss.backward()
    optimizer.step()
    
    # Training history
    history["total_loss"].append(float(total_loss.item()))
    history["loss_data"].append(float(loss_data_value.item()))
    history["loss_data_u"].append(float(loss_u.item()))
    history["loss_data_v"].append(float(loss_v.item()))
    history["loss_data_w"].append(float(loss_w.item()))
    history["loss_div"].append(float(loss_div.item()))
    history["lambda_data_u"].append(lambda_data_u)
    history["lambda_data_v"].append(lambda_data_v)
    history["lambda_data_w"].append(lambda_data_w)
    history["lambda_div"].append(lambda_div)
    history["lr"].append(
        float(optimizer.param_groups[0]["lr"])
    )

    # Validation and best checkpoint
    if epoch % eval_every == 0:
        val_metrics = evaluate_velocity_validation(
            model=net,
            X_val=X_val_data,
            device=device,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            batch_size=validation_batch_size,
        )
        # Divergence threshold: set only once at the end of the warmup period.
        if epoch >= warmup_epochs and max_divergence_threshold is None:
            max_divergence_threshold = (
                val_metrics["rms_divergence"] * divergence_multiplier
            )
        if max_divergence_threshold is None:
            divergence_constraint_satisfied = False
            selection_score = float("nan")
        else:
            divergence_constraint_satisfied = (
                val_metrics["rms_divergence"] <= max_divergence_threshold
            )
            selection_score = (
                val_metrics["relative_l2_velocity_vector"]
                if divergence_constraint_satisfied
                else float("inf")
            )
        val_metrics.update({
            "divergence_constraint_satisfied": bool(divergence_constraint_satisfied),
            "max_divergence_threshold": (
                float(max_divergence_threshold)
                if max_divergence_threshold is not None
                else float("nan")
            ),
            "selection_score": float(selection_score),
        })
        stopping_result = early_stopping.step(
            metric=val_metrics["selection_score"],
            epoch=epoch,
            model=net,
            optimizer=optimizer,
            extra_state={
                "history": history,
                "validation_metrics": val_metrics,
                "validation_history": validation_history,
                "max_divergence_threshold": max_divergence_threshold,
                "divergence_multiplier": divergence_multiplier,
                "dic_adim_norm": dic_adim_norm,
                "loss_weights": loss_weights,
                "lambda_div": lambda_div,
                "training_seed": training_seed,
                "split_seed": split_seed,
                "batch_seed": batch_seed,
                "L": L,
                "T": T,
                "U": U,
                "rho": rho,
                "mu": mu,
                "p0": p0,
                "Re": Re,
            },
        )
        validation_row = {
            "epoch": epoch,
            **val_metrics,
            "improved": stopping_result.improved,
            "best_epoch": stopping_result.best_epoch,
            "best_metric": stopping_result.best_metric,
            "evaluations_without_improvement": (
                stopping_result.evaluations_without_improvement
            ),
        }
        validation_history.append(validation_row)
        with validation_csv_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=validation_fields,
            )
            writer.writerow(validation_row)
        elapsed = time.time() - last_print_time
        print("\n" + "=" * 90)
        print(f"Epoch:                         {epoch}")
        print(f"Elapsed since last eval:       {elapsed:.2f} s")
        print(f"Training total loss:           {total_loss.item():.6e}")
        print(f"Validation vector relative L2: {100.0 * val_metrics['relative_l2_velocity_vector']:.4f} %")
        print(f"Validation RMS divergence:     {val_metrics['rms_divergence']:.6e}")
        print(f"Max divergence threshold:      {val_metrics['max_divergence_threshold']:.6e}")
        print(f"Divergence constraint met:     {val_metrics['divergence_constraint_satisfied']}")
        print(f"Selection score:               {val_metrics['selection_score']:.6e}")
        print(f"Best epoch:                    {stopping_result.best_epoch}")
        print(f"Best selection score:          {stopping_result.best_metric:.6e}")
        print(f"No-improvement evaluations:    {stopping_result.evaluations_without_improvement}/{patience}")
        print(f"Checkpoint improved:           {stopping_result.improved}")
        print("=" * 90)
        last_print_time = time.time()
        if stopping_result.should_stop:
            print(
                f"\nEarly stopping activated at epoch {epoch}. "
                f"Best epoch: {stopping_result.best_epoch}."
            )
            break

    
    # Save regular checkpoints
    if epoch % save_regular_every == 0:
        regular_checkpoint = {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "validation_history": validation_history,
            "max_divergence_threshold": max_divergence_threshold,
            "divergence_multiplier": divergence_multiplier,
            "dic_adim_norm": dic_adim_norm,
            "loss_weights": loss_weights,
            "lambda_div": lambda_div,
            "training_seed": training_seed,
            "split_seed": split_seed,
            "batch_seed": batch_seed,
            "L": L,
            "T": T,
            "U": U,
            "rho": rho,
            "mu": mu,
            "p0": p0,
            "Re": Re,
        }
        torch.save(
            regular_checkpoint,
            checkpoints_dir / f"checkpoint_{epoch}.pth",
        )

##########################################################
# Save 

# Save final state
last_checkpoint = {
    "epoch": epoch,
    "model_state_dict": net.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "history": history,
    "validation_history": validation_history,
    "max_divergence_threshold": max_divergence_threshold,
    "divergence_multiplier": divergence_multiplier,
    "dic_adim_norm": dic_adim_norm,
    "loss_weights": loss_weights,
    "lambda_div": lambda_div,
    "training_seed": training_seed,
    "split_seed": split_seed,
    "batch_seed": batch_seed,
    "best_epoch": early_stopping.best_epoch,
    "best_validation_metric": early_stopping.best_metric,
    "L": L,
    "T": T,
    "U": U,
    "rho": rho,
    "mu": mu,
    "p0": p0,
    "Re": Re,
}
torch.save(
    last_checkpoint,
    last_checkpoint_path,
)

# Save training summary
summary = {
    "experiment_name": experiment_name,
    "data_path": str(data_path),
    "output_dir": str(base_dir),
    "training_seed": training_seed,
    "split_seed": split_seed,
    "batch_seed": batch_seed,
    "last_epoch": int(epoch),
    "lambda_div": lambda_div,
    "divergence_multiplier": divergence_multiplier,
    "max_divergence_threshold": (
        float(max_divergence_threshold)
        if max_divergence_threshold is not None
        else None
    ),
    "best_epoch": int(early_stopping.best_epoch),
    "best_validation_selection_score": float(
        early_stopping.best_metric
    ),
    "best_checkpoint": str(best_checkpoint_path),
    "last_checkpoint": str(last_checkpoint_path),
}
save_json(
    summary,
    summary_json_path,
)
print("\nTraining finished successfully.")
print(f"Best epoch:       {early_stopping.best_epoch}")
print(
    "Best selection score: "
    f"{early_stopping.best_metric:.6e}"
)
print(f"Best checkpoint:  {best_checkpoint_path}")
print(f"Last checkpoint:  {last_checkpoint_path}")

