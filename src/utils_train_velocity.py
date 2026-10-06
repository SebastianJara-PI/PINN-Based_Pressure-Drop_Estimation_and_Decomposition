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


# Patience only accumulates from warmup_epochs.
# Improvement: metric < best_metric - min_delta_relative * |best_metric|,
# with numerical protection for metrics close to zero.
class EarlyStopping:

    def __init__(
        self, checkpoint_path, patience, warmup_epochs,
        min_delta_relative=1e-3,
    ):
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
        required_delta = self.min_delta_relative * max(abs(self.best_metric), 1e-12)
        return metric < self.best_metric - required_delta

    def step(self, metric, epoch, model, optimizer, scheduler=None, extra_state=None):
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
            evaluations_without_improvement=self.evaluations_without_improvement,
        )


# This function evaluates the velocity and divergence on the validation set.
def evaluate_velocity_validation(
    model, X_val, device, sigma_x=1.0, sigma_y=1.0, sigma_z=1.0,
    batch_size=20_000, eps=1e-12,
):
    if X_val.ndim != 2 or X_val.shape[1] < 7:
        raise ValueError("X_val must have at least 7 columns: [x,y,z,t,u,v,w].")

    model.eval()
    pred_chunks = []
    true_chunks = []
    divergence_chunks = []

    for start in range(0, len(X_val), batch_size):
        stop = min(start + batch_size, len(X_val))
        batch = X_val[start:stop]
        X_tensor = torch.tensor(
            batch, dtype=torch.float64, device=device
        ).requires_grad_(True)

        x = X_tensor[:, 0:1]
        y = X_tensor[:, 1:2]
        z = X_tensor[:, 2:3]
        t = X_tensor[:, 3:4]


        with torch.enable_grad():
            pred = model(x, y, z, t)[:, 0:3]
            u_x = torch.autograd.grad(
                pred[:, 0:1], x,
                grad_outputs=torch.ones_like(pred[:, 0:1]),
                retain_graph=True,
            )[0] / sigma_x
            v_y = torch.autograd.grad(
                pred[:, 1:2], y,
                grad_outputs=torch.ones_like(pred[:, 1:2]),
                retain_graph=True,
            )[0] / sigma_y
            w_z = torch.autograd.grad(
                pred[:, 2:3], z,
                grad_outputs=torch.ones_like(pred[:, 2:3]),
            )[0] / sigma_z
            divergence = u_x + v_y + w_z

        true = X_tensor[:, 4:7]
        pred_chunks.append(pred.detach().cpu().numpy())
        true_chunks.append(true.detach().cpu().numpy())
        divergence_chunks.append(divergence.detach().cpu().numpy())

    velocity_pred = np.concatenate(pred_chunks, axis=0)
    velocity_true = np.concatenate(true_chunks, axis=0)
    divergence_pred = np.concatenate(divergence_chunks, axis=0)

    speed_pred = np.linalg.norm(velocity_pred, axis=1)
    speed_true = np.linalg.norm(velocity_true, axis=1)
    error_speed = speed_pred - speed_true

    rmse_speed = float(np.sqrt(np.mean(error_speed**2)))
    speed_range = float(np.ptp(speed_true))
    nrmse_range_speed = rmse_speed / max(speed_range, eps)
    relative_l2_speed = float(
        np.linalg.norm(error_speed) / max(np.linalg.norm(speed_true), eps)
    )
    relative_l2_velocity_vector = float(
        np.linalg.norm(velocity_pred - velocity_true)
        / max(np.linalg.norm(velocity_true), eps)
    )
    rms_divergence = float(np.sqrt(np.mean(divergence_pred**2)))

    mse_u = float(np.mean((velocity_pred[:, 0] - velocity_true[:, 0])**2))
    mse_v = float(np.mean((velocity_pred[:, 1] - velocity_true[:, 1])**2))
    mse_w = float(np.mean((velocity_pred[:, 2] - velocity_true[:, 2])**2))

    metrics = {
        "rmse_speed": rmse_speed,
        "nrmse_range_speed": float(nrmse_range_speed),
        "relative_l2_speed": relative_l2_speed,
        "relative_l2_velocity_vector": relative_l2_velocity_vector,
        "rms_divergence": rms_divergence,
        "mse_u": mse_u,
        "mse_v": mse_v,
        "mse_w": mse_w,
    }

    model.train()
    return metrics


# Saves a dictionary as a JSON file.
def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)









