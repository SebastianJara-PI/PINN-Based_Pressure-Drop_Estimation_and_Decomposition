import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

# Fix seeds for reproducibility
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Early stopping implementation for training neural networks
class EarlyStoppingResult:

    def __init__(
        self, improved, should_stop, best_metric, best_epoch,
        evaluations_without_improvement,
    ):
        self.improved = improved
        self.should_stop = should_stop
        self.best_metric = best_metric
        self.best_epoch = best_epoch
        self.evaluations_without_improvement = evaluations_without_improvement


class EarlyStopping:

    def __init__(self, checkpoint_path, patience=10, warmup_epochs=10000,
                 min_delta_relative=1e-3):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.patience = int(patience)
        self.warmup_epochs = int(warmup_epochs)
        self.min_delta_relative = float(min_delta_relative)
        self.best_metric = float("inf")
        self.best_epoch = -1
        self.evaluations_without_improvement = 0

    def _is_improvement(self, metric):
        if not np.isfinite(metric):
            return False
        if self.best_metric == float("inf"):
            return True
        required_delta = self.min_delta_relative * max(
            abs(self.best_metric), 1e-12
        )
        return metric < self.best_metric - required_delta

    def step(self, metric, epoch, model, optimizer, scheduler=None,
             extra_state=None):
        improved = self._is_improvement(metric)
        if improved:
            self.best_metric = float(metric)
            self.best_epoch = int(epoch)
            self.evaluations_without_improvement = 0
            checkpoint = {
                "epoch": int(epoch),
                "validation_metric": float(metric),
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": optimizer.state_dict(),
                "extra_state": extra_state or {},
            }
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(checkpoint, self.checkpoint_path)
        elif epoch >= self.warmup_epochs:
            self.evaluations_without_improvement += 1

        should_stop = (
            epoch >= self.warmup_epochs
            and self.evaluations_without_improvement >= self.patience
        )
        return EarlyStoppingResult(
            improved=improved,
            should_stop=should_stop,
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            evaluations_without_improvement=(
                self.evaluations_without_improvement
            ),
        )

# Build inputs for pressure drop evaluation
def build_pressure_drop_inputs(centerline_xyz, times, idx_p1, idx_p2,
                               dic_adim_norm, L, T):
    centerline_xyz = np.asarray(centerline_xyz, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    endpoints = centerline_xyz[[idx_p1, idx_p2], :]
    mu = np.array([
        dic_adim_norm["mu_x"],
        dic_adim_norm["mu_y"],
        dic_adim_norm["mu_z"],
    ], dtype=np.float64)
    sigma = np.array([
        dic_adim_norm["sigma_x"],
        dic_adim_norm["sigma_y"],
        dic_adim_norm["sigma_z"],
    ], dtype=np.float64)
    xyz_norm = (endpoints / float(L) - mu) / sigma
    time_norm = (
        times / float(T) - float(dic_adim_norm["mu_t"])
    ) / float(dic_adim_norm["sigma_t"])
    X_drop = np.empty((len(times), 2, 4), dtype=np.float64)
    X_drop[:, :, :3] = xyz_norm[None, :, :]
    X_drop[:, :, 3] = time_norm[:, None]
    return X_drop, endpoints


# This function associates each data point with the closest tangent of the P1-P2 segment
def build_centerline_tangents_for_points(
    centerline_xyz,
    data_xyz,
    idx_p1,
    idx_p2,
    eps=1e-12,
):
    
    centerline_xyz = np.asarray(centerline_xyz, dtype=np.float64)
    data_xyz = np.asarray(data_xyz, dtype=np.float64)
    if idx_p1 == idx_p2:
        raise ValueError("P1 and P2 must be distinct points")
    if idx_p1 < idx_p2:
        path_indices = np.arange(idx_p1, idx_p2 + 1)
    else:
        path_indices = np.arange(idx_p1, idx_p2 - 1, -1)
    path_xyz = centerline_xyz[path_indices]
    if len(path_xyz) < 2:
        raise ValueError("The P1-P2 segment requires at least two points")

    tangents = np.gradient(path_xyz, axis=0)
    tangent_norm = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(tangent_norm, eps)

    distances = np.linalg.norm(
        data_xyz[:, None, :] - path_xyz[None, :, :],
        axis=2,
    )
    nearest_local_index = np.argmin(distances, axis=1)
    nearest_distance = distances[
        np.arange(len(data_xyz)), nearest_local_index
    ]
    return (
        tangents[nearest_local_index],
        nearest_distance,
        path_indices[nearest_local_index],
    )


def _grad(p, x):
    return torch.autograd.grad(
        p,
        x,
        grad_outputs=torch.ones_like(p),
        create_graph=False,
        retain_graph=True,
    )[0]


# This function evaluates the pressure validation without using clinical pressure.
def evaluate_pressure_validation(
    model,
    X_val,
    Y_val,
    tangent_val,
    X_drop,
    drop_phase,
    pressure_scale_mmhg,
    device,
    sigma_x,
    sigma_y,
    sigma_z,
    score_weights=(0.75, 0.25),
    batch_size=10000,
    eps=1e-12,
    return_curves=False,
):
    model.eval()
    pred_parts = [[], [], []]
    for start in range(0, len(X_val), batch_size):
        stop = min(start + batch_size, len(X_val))
        X = torch.tensor(
            X_val[start:stop],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        x, y, z, t = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]
        with torch.enable_grad():
            out = model(x, y, z, t)
            for component in range(3):
                p = out[:, component:component + 1]
                gradient = torch.cat([
                    _grad(p, x) / sigma_x,
                    _grad(p, y) / sigma_y,
                    _grad(p, z) / sigma_z,
                ], dim=1)
                pred_parts[component].append(
                    gradient.detach().cpu().numpy()
                )

    preds = [np.concatenate(values, axis=0) for values in pred_parts]
    refs = [Y_val[:, 0:3], Y_val[:, 3:6], Y_val[:, 6:9]]
    tangent_val = np.asarray(tangent_val, dtype=np.float64)
    if tangent_val.shape != (len(X_val), 3):
        raise ValueError(
            "tangent_val must have shape (len(X_val), 3)"
        )

    result = {}
    component_relative_mses = []
    for name, pred, ref in zip(["acc", "adv", "vis"], preds, refs):
        error = pred - ref
        error_energy = float(np.sum(error**2))
        reference_energy = float(np.sum(ref**2))
        mse_vec = float(np.mean(np.sum(error**2, axis=1)))
        ref_energy = float(np.mean(np.sum(ref**2, axis=1)))
        rmse = float(np.sqrt(np.mean(error**2)))
        relative_mse = error_energy / max(reference_energy, eps)
        component_relative_mses.append(relative_mse)
        result[f"relative_mse_{name}"] = relative_mse
        result[f"relative_l2_{name}"] = float(
            np.linalg.norm(error) / max(np.linalg.norm(ref), eps)
        )
        result[f"nrmse_range_{name}"] = (
            rmse / max(float(np.ptp(ref)), eps)
        )
        result[f"mse_{name}"] = mse_vec
        result[f"reference_energy_{name}"] = ref_energy

    total_pred = np.sum(np.stack(preds, axis=0), axis=0)
    total_ref = np.sum(np.stack(refs, axis=0), axis=0)
    total_error = total_pred - total_ref
    total_vector_relative_mse = (
        float(np.sum(total_error**2))
        / max(float(np.sum(total_ref**2)), eps)
    )
    predicted_parallel = np.sum(total_pred * tangent_val, axis=1)
    reference_parallel = np.sum(total_ref * tangent_val, axis=1)
    tangential_relative_mse = (
        float(np.sum((predicted_parallel - reference_parallel) ** 2))
        / max(float(np.sum(reference_parallel**2)), eps)
    )
    component_balanced_relative_mse = float(
        np.mean(component_relative_mses)
    )

    w_tangent, w_vector = map(float, score_weights)
    if not np.isclose(w_tangent + w_vector, 1.0):
        raise ValueError("The score weights must sum to one")
    pressure_gradient_score = (
        w_tangent * tangential_relative_mse
        + w_vector * total_vector_relative_mse
    )
    result.update({
        "gradient_tangential_relative_mse": tangential_relative_mse,
        "gradient_total_vector_relative_mse": total_vector_relative_mse,
        "gradient_component_balanced_relative_mse": (
            component_balanced_relative_mse
        ),
        "pressure_gradient_score": float(pressure_gradient_score),
    })

    X_drop_flat = np.asarray(X_drop, dtype=np.float64).reshape(-1, 4)
    with torch.no_grad():
        X_tensor = torch.tensor(
            X_drop_flat,
            dtype=torch.float64,
            device=device,
        )
        endpoint_pressure = model(
            X_tensor[:, 0:1],
            X_tensor[:, 1:2],
            X_tensor[:, 2:3],
            X_tensor[:, 3:4],
        ).detach().cpu().numpy().reshape(-1, 2, 3)

    component_drop = float(pressure_scale_mmhg) * (
        endpoint_pressure[:, 0, :] - endpoint_pressure[:, 1, :]
    )
    total_drop = np.sum(component_drop, axis=1)
    phase = np.asarray(drop_phase, dtype=np.float64)
    if len(phase) != len(total_drop):
        raise ValueError("The phase and the drop curve do not match")
    result.update({
        "drop_rms_mmhg": float(np.sqrt(np.mean(total_drop**2))),
        "drop_peak_to_peak_mmhg": float(np.ptp(total_drop)),
        "drop_max_abs_mmhg": float(np.max(np.abs(total_drop))),
    })
    curves = {
        "phase": phase.copy(),
        "drop_total_mmhg": total_drop.copy(),
        "drop_acc_mmhg": component_drop[:, 0].copy(),
        "drop_adv_mmhg": component_drop[:, 1].copy(),
        "drop_vis_mmhg": component_drop[:, 2].copy(),
    }
    model.train()
    if return_curves:
        return result, curves
    return result


# Relative L2 change between two successive curves
def relative_curve_change(current_curve, previous_curve, eps=1e-12):
    if previous_curve is None:
        return float("inf")
    current_curve = np.asarray(current_curve, dtype=np.float64)
    previous_curve = np.asarray(previous_curve, dtype=np.float64)
    if current_curve.shape != previous_curve.shape:
        raise ValueError("The successive curves must have the same shape")
    return float(
        np.linalg.norm(current_curve - previous_curve)
        / max(np.linalg.norm(previous_curve), eps)
    )


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
