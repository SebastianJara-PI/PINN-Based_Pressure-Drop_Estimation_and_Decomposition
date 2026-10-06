
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

from loss_functions import pressure_gradient_loss
from loss_weights import update_loss_weights_pressure
from utils import prepare_pressure_training_data, train_test_split_numpy
from NN import MLP
from utils_train_pressure import (
    EarlyStopping,
    build_centerline_tangents_for_points,
    build_pressure_drop_inputs,
    evaluate_pressure_validation,
    relative_curve_change,
    save_json,
    set_all_seeds,
)

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##########################################################
# Case and experiment configuration

# Case configuration
CASE_NAME = "cfd_9mm"
VELOCITY_RUN_NAME = "seed_001"
experiment_name = CASE_NAME

training_seed = 1
split_seed = 43
batch_seed = 20000 + training_seed
max_epochs = 100000 
eval_every = 500
save_regular_every = 2500
warmup_epochs = 5000
patience = 15
min_delta_relative = 2e-3
drop_stability_tolerance = 1e-2
drop_stability_required = 3
score_weights = (0.75, 0.25)
training_batch_size = 4000
validation_batch_size = 10000

set_all_seeds(training_seed)

##########################################################
# Paths
data_dir = PROJECT_ROOT / "data" / CASE_NAME / "processed"
velocity_run_dir = (
    PROJECT_ROOT / "outputs" / CASE_NAME / "velocity" / VELOCITY_RUN_NAME
)
velocity_ckpt_path = velocity_run_dir / "checkpoints" / "best_velocity.pth"
derived_dir = (
    PROJECT_ROOT / "outputs" / CASE_NAME / "momentum_terms" / VELOCITY_RUN_NAME
)

pressure_results_dir = PROJECT_ROOT / "outputs" / CASE_NAME / "pressure"
run_name = f"seed_{training_seed:03d}"
base_dir = pressure_results_dir / run_name

checkpoints_dir = base_dir / "checkpoints"
sampling_dir = base_dir / "sampling"
logs_dir = base_dir / "logs"
for p in [checkpoints_dir, sampling_dir, logs_dir]:
    p.mkdir(parents=True, exist_ok=True)

best_ckpt_path = checkpoints_dir / "best_pressure.pth"
last_ckpt_path = checkpoints_dir / "checkpoint_last.pth"
csv_path = logs_dir / "validation_history.csv"
summary_path = logs_dir / "training_summary.json"

##########################################################
# Velocity metadata
if not velocity_ckpt_path.is_file():
    raise FileNotFoundError(f"No existe el checkpoint de velocidad: {velocity_ckpt_path}")
if not (derived_dir / "Data_acc_mks.npy").is_file():
    raise FileNotFoundError(f"No existen los términos reconstruidos en: {derived_dir}")
centerline_path = data_dir / "centerline_xyz.npy"
if not centerline_path.is_file():
    raise FileNotFoundError(
        f"No existe la centerline: {centerline_path}. "
        "Ejecuta primero 01_prepare_centerline_data.py."
    )

vel_ckpt = torch.load(velocity_ckpt_path, map_location="cpu")
extra = vel_ckpt["extra_state"]
if int(extra["training_seed"]) != training_seed:
    raise ValueError("La semilla del checkpoint no corresponde a training_seed.")

dic_adim_norm = extra["dic_adim_norm"]
L, T, U = float(extra["L"]), float(extra["T"]), float(extra["U"])
rho, mu = float(extra["rho"]), float(extra["mu"])
p0, Re = float(extra["p0"]), float(extra["Re"])

sigma_x = float(dic_adim_norm["sigma_x"])
sigma_y = float(dic_adim_norm["sigma_y"])
sigma_z = float(dic_adim_norm["sigma_z"])

##########################################################
# Derived data
Data_acc_mks = np.load(derived_dir / "Data_acc_mks.npy", allow_pickle=False)
Data_adv_mks = np.load(derived_dir / "Data_adv_mks.npy", allow_pickle=False)
Data_visc_mks = np.load(derived_dir / "Data_visc_mks.npy", allow_pickle=False)

X_all, Y_all = prepare_pressure_training_data(
    Data_acc_mks=Data_acc_mks,
    Data_adv_mks=Data_adv_mks,
    Data_visc_mks=Data_visc_mks,
    dic_adim_norm=dic_adim_norm,
    L=L, T=T, U=U,
)

##########################################################
# Geometry P1-P2 and tangents for the non-clinical score
centerline_xyz = np.load(centerline_path, allow_pickle=False)

# Fixed anatomical points used for the stable drop.
idx_p1 = 0
idx_p2 = centerline_xyz.shape[0] - 1

model_times = np.asarray(Data_acc_mks[:, 0, 3], dtype=np.float64)
model_phase = (
    (model_times - model_times[0])
    / (model_times[-1] - model_times[0])
)

X_drop, pressure_endpoints = build_pressure_drop_inputs(
    centerline_xyz=centerline_xyz,
    times=model_times,
    idx_p1=idx_p1,
    idx_p2=idx_p2,
    dic_adim_norm=dic_adim_norm,
    L=L,
    T=T,
)
spatial_xyz = np.asarray(Data_acc_mks[0, :, 0:3], dtype=np.float64)
(
    tangent_space,
    centerline_nearest_distance,
    centerline_nearest_index,
) = build_centerline_tangents_for_points(
    centerline_xyz=centerline_xyz,
    data_xyz=spatial_xyz,
    idx_p1=idx_p1,
    idx_p2=idx_p2,
)
tangent_all = np.tile(tangent_space, (len(model_times), 1))
pressure_scale_mmhg = p0 / 133.322

##########################################################
# Fixed train/validation split
X_train, X_val, idx_train, idx_val = train_test_split_numpy(
    X_all,
    train_ratio=0.90,
    seed=split_seed,
)
Y_train, Y_val = Y_all[idx_train], Y_all[idx_val]
tangent_val = tangent_all[idx_val]

np.savez(
    sampling_dir / "sampling_index.npz",
    idx_train_data=idx_train,
    idx_val_data=idx_val,
    split_seed=split_seed,
)

np.savez(
    sampling_dir / "dic_adim_norm.npz",
    **dic_adim_norm,
    L=L, T=T, U=U, rho=rho, mu=mu, p0=p0, Re=Re,
)

np.savez(
    sampling_dir / "pressure_validation_geometry.npz",
    model_times=model_times,
    model_phase=model_phase,
    X_drop=X_drop,
    pressure_endpoints=pressure_endpoints,
    tangent_space=tangent_space,
    tangent_val=tangent_val,
    centerline_nearest_distance=centerline_nearest_distance,
    centerline_nearest_index=centerline_nearest_index,
    idx_p1=idx_p1,
    idx_p2=idx_p2,
    pressure_scale_mmhg=pressure_scale_mmhg,
    score_weights=np.asarray(score_weights),
    drop_stability_tolerance=drop_stability_tolerance,
)

print("=" * 90)
print("Device:", device)
print("Velocity run:", VELOCITY_RUN_NAME)
print("Derived terms:", derived_dir)
print("Pressure results:", base_dir)
print("Train/val:", len(X_train), len(X_val))
print("Drop endpoints P1/P2:", pressure_endpoints[0], pressure_endpoints[1])
print("Model samples:", len(model_phase))
print("Nearest centerline distance [mm], median/max:",
      1e3 * np.median(centerline_nearest_distance),
      1e3 * np.max(centerline_nearest_distance))
print("Pressure scale [mmHg]:", pressure_scale_mmhg)
print("=" * 90)


# Model
net = MLP(
    input_size=4,
    output_size=3,
    hidden_layers=3,
    hidden_units=32,
    activation_fn=nn.SiLU(),
).to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=350000,
    gamma=0.1,
)

loss_weights = np.array([1.0, 1.0, 1.0], dtype=np.float64)
rng = np.random.default_rng(batch_seed)

early_stopping = EarlyStopping(
    checkpoint_path=best_ckpt_path,
    patience=patience,
    warmup_epochs=warmup_epochs,
    min_delta_relative=min_delta_relative,
)

history = {
    "total_loss": [],
    "loss_acc": [],
    "loss_adv": [],
    "loss_vis": [],
    "lambda_acc": [],
    "lambda_adv": [],
    "lambda_vis": [],
    "lr": [],
}
validation_history = []
previous_drop_curve = None
drop_stability_consecutive = 0

fields = [
    "epoch", "pressure_selection_metric", "pressure_gradient_score",
    "gradient_tangential_relative_mse",
    "gradient_total_vector_relative_mse",
    "gradient_component_balanced_relative_mse",
    "drop_stability_relative", "drop_stability_satisfied",
    "drop_stability_consecutive",
    "drop_rms_mmhg", "drop_peak_to_peak_mmhg", "drop_max_abs_mmhg",
    "relative_mse_acc", "relative_mse_adv", "relative_mse_vis",
    "relative_l2_acc", "relative_l2_adv", "relative_l2_vis",
    "nrmse_range_acc", "nrmse_range_adv", "nrmse_range_vis",
    "mse_acc", "mse_adv", "mse_vis",
    "reference_energy_acc", "reference_energy_adv", "reference_energy_vis",
    "improved", "best_epoch", "best_metric",
    "evaluations_without_improvement", "early_stop_ready",
]

with csv_path.open("w", newline="", encoding="utf-8") as f:
    csv.DictWriter(f, fieldnames=fields).writeheader()

last_print = time.time()

##########################################################
# Training

for epoch in range(max_epochs + 1):
    n_b = min(training_batch_size, len(X_train))
    batch_idx = rng.choice(len(X_train), size=n_b, replace=False)

    Xb = torch.tensor(X_train[batch_idx], dtype=torch.float64, device=device, requires_grad=True)
    Yb = torch.tensor(Y_train[batch_idx], dtype=torch.float64, device=device)

    if epoch % 100 == 0:
        loss_weights, _ = update_loss_weights_pressure(
            model=net,
            loss_weights=loss_weights,
            X_batch_tensor=Xb,
            Y_grad_batch_tensor=Yb,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            alpha=0.8,
            lambda_min=1e-3,
            lambda_max=1e3,
        )

    lambda_acc, lambda_adv, lambda_vis = map(float, loss_weights)

    optimizer.zero_grad(set_to_none=True)
    Xb = Xb.detach().clone().requires_grad_(True)

    loss_acc, loss_adv, loss_vis = pressure_gradient_loss(
        model=net,
        X_tensor_norm=Xb,
        Y_grad_tensor=Yb,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        sigma_z=sigma_z,
    )

    total_loss = (
        lambda_acc * loss_acc
        + lambda_adv * loss_adv
        + lambda_vis * loss_vis
    )

    total_loss.backward()
    optimizer.step()
    scheduler.step()

    history["total_loss"].append(float(total_loss.item()))
    history["loss_acc"].append(float(loss_acc.item()))
    history["loss_adv"].append(float(loss_adv.item()))
    history["loss_vis"].append(float(loss_vis.item()))
    history["lambda_acc"].append(lambda_acc)
    history["lambda_adv"].append(lambda_adv)
    history["lambda_vis"].append(lambda_vis)
    history["lr"].append(float(optimizer.param_groups[0]["lr"]))

    if epoch % eval_every == 0:
        metrics, drop_curves = evaluate_pressure_validation(
            model=net,
            X_val=X_val,
            Y_val=Y_val,
            tangent_val=tangent_val,
            X_drop=X_drop,
            drop_phase=model_phase,
            pressure_scale_mmhg=pressure_scale_mmhg,
            device=device,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            sigma_z=sigma_z,
            score_weights=score_weights,
            batch_size=validation_batch_size,
            return_curves=True,
        )

        drop_stability_relative = relative_curve_change(
            current_curve=drop_curves["drop_total_mmhg"],
            previous_curve=previous_drop_curve,
        )
        drop_stability_satisfied = (
            epoch >= warmup_epochs
            and drop_stability_relative < drop_stability_tolerance
        )
        if drop_stability_satisfied:
            drop_stability_consecutive += 1
        else:
            drop_stability_consecutive = 0
        previous_drop_curve = drop_curves["drop_total_mmhg"].copy()

        selection_metric = (
            metrics["pressure_gradient_score"]
            if epoch >= warmup_epochs
            else float("nan")
        )
        metrics.update({
            "drop_stability_relative": drop_stability_relative,
            "drop_stability_satisfied": drop_stability_satisfied,
            "drop_stability_consecutive": drop_stability_consecutive,
            "pressure_selection_metric": selection_metric,
        })

        result = early_stopping.step(
            metric=selection_metric,
            epoch=epoch,
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            extra_state={
                "history": history,
                "validation_history": validation_history,
                "validation_metrics": metrics,
                "pressure_drop_curves": drop_curves,
                "dic_adim_norm": dic_adim_norm,
                "loss_weights": loss_weights,
                "training_seed": training_seed,
                "split_seed": split_seed,
                "batch_seed": batch_seed,
                "velocity_run_name": VELOCITY_RUN_NAME,
                "velocity_checkpoint": str(velocity_ckpt_path),
                "derived_terms_directory": str(derived_dir),
                "idx_p1": idx_p1,
                "idx_p2": idx_p2,
                "pressure_endpoints": pressure_endpoints,
                "pressure_scale_mmhg": pressure_scale_mmhg,
                "score_weights": {
                    "tangential": score_weights[0],
                    "total_vector": score_weights[1],
                },
                "drop_stability_tolerance": drop_stability_tolerance,
                "drop_stability_required": drop_stability_required,
                "drop_stability_consecutive": drop_stability_consecutive,
                "L": L, "T": T, "U": U,
                "rho": rho, "mu": mu, "p0": p0, "Re": Re,
            },
        )

        early_stop_ready = (
            result.should_stop
            and drop_stability_consecutive >= drop_stability_required
        )
        row = {
            "epoch": epoch,
            **metrics,
            "improved": result.improved,
            "best_epoch": result.best_epoch,
            "best_metric": result.best_metric,
            "evaluations_without_improvement": result.evaluations_without_improvement,
            "early_stop_ready": early_stop_ready,
        }
        validation_history.append(row)

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        print("\n" + "=" * 90)
        print("Epoch:", epoch)
        print("Elapsed:", f"{time.time()-last_print:.2f} s")
        print("Gradient score:", f"{metrics['pressure_gradient_score']:.6e}")
        print("Tangential/vector/balanced:",
              f"{metrics['gradient_tangential_relative_mse']:.4e}",
              f"{metrics['gradient_total_vector_relative_mse']:.4e}",
              f"{metrics['gradient_component_balanced_relative_mse']:.4e}")
        print("Drop stability:", f"{drop_stability_relative:.4e}",
              f"({drop_stability_consecutive}/{drop_stability_required})")
        print("Drop RMS/peak-to-peak [mmHg]:",
              f"{metrics['drop_rms_mmhg']:.4f}",
              f"{metrics['drop_peak_to_peak_mmhg']:.4f}")
        print("Rel MSE acc/adv/vis:",
              f"{metrics['relative_mse_acc']:.4e}",
              f"{metrics['relative_mse_adv']:.4e}",
              f"{metrics['relative_mse_vis']:.4e}")
        print("Best epoch:", result.best_epoch)
        print("No improvement:",
              f"{result.evaluations_without_improvement}/{patience}")
        print("Early-stop ready:", early_stop_ready)
        print("=" * 90)
        last_print = time.time()

        if early_stop_ready:
            print(f"Early stopping at {epoch}; best epoch {result.best_epoch}")
            break

    if epoch % save_regular_every == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "validation_history": validation_history,
            "dic_adim_norm": dic_adim_norm,
            "loss_weights": loss_weights,
            "training_seed": training_seed,
            "split_seed": split_seed,
            "batch_seed": batch_seed,
            "velocity_run_name": VELOCITY_RUN_NAME,
            "idx_p1": idx_p1,
            "idx_p2": idx_p2,
            "pressure_endpoints": pressure_endpoints,
            "pressure_scale_mmhg": pressure_scale_mmhg,
            "score_weights": score_weights,
            "drop_stability_tolerance": drop_stability_tolerance,
            "drop_stability_required": drop_stability_required,
            "drop_stability_consecutive": drop_stability_consecutive,
            "previous_drop_curve": previous_drop_curve,
            "L": L, "T": T, "U": U,
            "rho": rho, "mu": mu, "p0": p0, "Re": Re,
        }, checkpoints_dir / f"checkpoint_{epoch}.pth")


##########################################################
# Final checkpoint save

torch.save({
    "epoch": epoch,
    "model_state_dict": net.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "history": history,
    "validation_history": validation_history,
    "dic_adim_norm": dic_adim_norm,
    "loss_weights": loss_weights,
    "best_epoch": early_stopping.best_epoch,
    "best_validation_metric": early_stopping.best_metric,
    "training_seed": training_seed,
    "split_seed": split_seed,
    "batch_seed": batch_seed,
    "velocity_run_name": VELOCITY_RUN_NAME,
    "idx_p1": idx_p1,
    "idx_p2": idx_p2,
    "pressure_endpoints": pressure_endpoints,
    "pressure_scale_mmhg": pressure_scale_mmhg,
    "score_weights": score_weights,
    "drop_stability_tolerance": drop_stability_tolerance,
    "drop_stability_required": drop_stability_required,
    "drop_stability_consecutive": drop_stability_consecutive,
    "previous_drop_curve": previous_drop_curve,
    "L": L, "T": T, "U": U,
    "rho": rho, "mu": mu, "p0": p0, "Re": Re,
}, last_ckpt_path)

save_json({
    "experiment_name": experiment_name,
    "training_seed": training_seed,
    "split_seed": split_seed,
    "batch_seed": batch_seed,
    "velocity_run_name": VELOCITY_RUN_NAME,
    "idx_p1": idx_p1,
    "idx_p2": idx_p2,
    "pressure_endpoints": pressure_endpoints.tolist(),
    "pressure_scale_mmhg": pressure_scale_mmhg,
    "warmup_epochs": warmup_epochs,
    "patience": patience,
    "min_delta_relative": min_delta_relative,
    "score_weights": {
        "tangential": score_weights[0],
        "total_vector": score_weights[1],
    },
    "drop_stability_tolerance": drop_stability_tolerance,
    "drop_stability_required": drop_stability_required,
    "last_epoch": int(epoch),
    "best_epoch": int(early_stopping.best_epoch),
    "best_pressure_validation_metric": float(early_stopping.best_metric),
    "best_checkpoint": str(best_ckpt_path),
    "last_checkpoint": str(last_ckpt_path),
}, summary_path)

print("Pressure training finished.")
print("Best epoch:", early_stopping.best_epoch)
print("Best metric:", early_stopping.best_metric)
print("Best checkpoint:", best_ckpt_path)
