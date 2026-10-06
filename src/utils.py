import numpy as np              
from scipy import ndimage as ndi
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.interpolate import splprep, splev


from skimage.morphology import skeletonize
from skimage.measure import label
import torch



def build_centerline_from_pointcloud(Data_grid, voxel_size, padding=2, smooth=1):
    """
    Estimate the centerline of a vessel from its segmentation point cloud.

    Voxelizes the points of the first time frame, closes small gaps, keeps
    the largest connected component, skeletonizes it in 3D, and takes the
    longest geodesic path between skeleton endpoints. The path is then
    smoothed with a spline.

    Parameters
    ----------
    Data_grid : array (Nt, Np, >=3)
        Columns x, y, z in physical coordinates. Only Data_grid[0] is used.
    voxel_size : float
        Voxel size in physical units (required).
    padding : int
        Number of voxels of margin around the point cloud.
    smooth : float
        Spline smoothing factor (0 = exact interpolation). If the fit
        fails, the unsmoothed path is returned.

    Returns
    -------
    centerline_xyz : array (M, 3)
        Centerline in physical coordinates, ordered from the path end
        to its start.
    skel_xyz : array (K, 3)
        Skeleton voxels in physical coordinates.
    mask : array 3D bool
        Voxelized binary mask.
    xyz_min : array (3,)
        Physical origin of the voxel grid.
    """

    pts = np.asarray(Data_grid[0, :, 0:3])

    # 1- Build voxel grid
    xyz_min = pts.min(axis=0) - padding * voxel_size
    xyz_max = pts.max(axis=0) + padding * voxel_size

    grid_shape = np.ceil((xyz_max - xyz_min) / voxel_size).astype(int) + 1
    nx, ny, nz = grid_shape

    mask = np.zeros((nx, ny, nz), dtype=bool)

    ijk = np.round((pts - xyz_min) / voxel_size).astype(int)
    ijk = np.clip(ijk, [0, 0, 0], grid_shape - 1)

    mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    # Optional: close small gaps
    mask = ndi.binary_closing(mask, structure=np.ones((3, 3, 3)))
    mask = ndi.binary_fill_holes(mask)

    # 2- Largest connected component
    lab = label(mask, connectivity=3)
    if lab.max() == 0:
        raise ValueError("The mask is empty after preprocessing.")

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    mask = (lab == largest)

    # 3- Skeleton 3D
    skel = skeletonize(mask) > 0

    # remove disconnected components
    skel_lab = label(skel, connectivity=3)
    if skel_lab.max() > 1:
        sizes = np.bincount(skel_lab.ravel())
        sizes[0] = 0
        largest = sizes.argmax()
        skel = (skel_lab == largest)

    skel_idx = np.argwhere(skel)
    if len(skel_idx) < 2:
        raise ValueError("The skeleton has too few points.")

    # 4- Build graph from the skeleton
    idx_map = {tuple(p): i for i, p in enumerate(skel_idx)}
    N = len(skel_idx)

    # connected neighbors
    offsets = []
    for a in [-1, 0, 1]:
        for b in [-1, 0, 1]:
            for c in [-1, 0, 1]:
                if (a, b, c) != (0, 0, 0):
                    offsets.append((a, b, c))

    G = lil_matrix((N, N), dtype=float)
    degree = np.zeros(N, dtype=int)

    for i, p in enumerate(skel_idx):
        px, py, pz = p
        for off in offsets:
            q = (px + off[0], py + off[1], pz + off[2])
            j = idx_map.get(q, None)
            if j is not None:
                w = np.linalg.norm(np.array(off, dtype=float))
                G[i, j] = w
                degree[i] += 1

    G = G.tocsr()

    # 5- find endpoints
    endpoints = np.where(degree == 1)[0]

    def farthest_node(start):
        dist = dijkstra(G, directed=False, indices=start)
        far = np.nanargmax(np.where(np.isfinite(dist), dist, -np.inf))
        return far, dist

    if len(endpoints) >= 2:
        # find the pair of endpoints with the greatest geodesic distance
        best_len = -np.inf
        best_pair = None
        best_pred = None

        for ep in endpoints:
            dist, pred = dijkstra(G, directed=False, indices=ep, return_predecessors=True)
            for ep2 in endpoints:
                if ep2 == ep:
                    continue
                if np.isfinite(dist[ep2]) and dist[ep2] > best_len:
                    best_len = dist[ep2]
                    best_pair = (ep, ep2)
                    best_pred = pred

        start, end = best_pair
        pred = best_pred
    else:
        # fallback: aproximación del diámetro del grafo
        a, _ = farthest_node(0)
        b, _ = farthest_node(a)
        dist, pred = dijkstra(G, directed=False, indices=a, return_predecessors=True)
        start, end = a, b

    # 6- reconstruct path
    path = [end]
    cur = end
    while cur != start and cur != -9999:
        cur = pred[cur]
        if cur != -9999:
            path.append(cur)

    ## change the path order to start from the start node
    # path = path[::-1]
    path_idx = skel_idx[path]

    # 7- convert to physical coordinates
    skel_xyz = xyz_min + skel_idx * voxel_size
    centerline_xyz = xyz_min + path_idx * voxel_size

    # 8- smooth the centerline with spline
    if len(centerline_xyz) >= 4:
        try:
            tck, u = splprep(
                [centerline_xyz[:, 0], centerline_xyz[:, 1], centerline_xyz[:, 2]],
                s=smooth
            )
            u_new = np.linspace(0, 1, max(200, len(centerline_xyz)))
            x_s, y_s, z_s = splev(u_new, tck)
            centerline_xyz = np.column_stack([x_s, y_s, z_s])
        except:
            pass

    return centerline_xyz, skel_xyz, mask, xyz_min



##################################################################
# Prepare data functions

def train_test_split_numpy(X, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    n_train = int(train_ratio * len(X))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    return X[train_idx], X[test_idx], train_idx, test_idx


def sample_batch(X, n_samples, rng):
    n_samples = min(n_samples, len(X))
    idx = rng.choice(len(X), size=n_samples, replace=False)
    return X[idx], idx

def normalize_dataset(
    X_raw,
    L,
    T,
    U,
    mu_x,
    mu_y,
    mu_z,
    mu_t,
    sigma_x,
    sigma_y,
    sigma_z,
    sigma_t,
):

    X_space_a = X_raw[:, 0:3] / L
    X_time_a = X_raw[:, 3:4] / T
    X_vel_a = X_raw[:, 4:7] / U

    x_norm = (X_space_a[:, 0:1] - mu_x) / sigma_x
    y_norm = (X_space_a[:, 1:2] - mu_y) / sigma_y
    z_norm = (X_space_a[:, 2:3] - mu_z) / sigma_z
    t_norm = (X_time_a[:, 0:1] - mu_t) / sigma_t

    X_norm = np.concatenate(
        [
            x_norm,
            y_norm,
            z_norm,
            t_norm,
            X_vel_a,
        ],
        axis=1,
    )

    return X_norm

def prepare_pressure_training_data(
    Data_acc_mks,
    Data_adv_mks,
    Data_visc_mks,
    dic_adim_norm,
    L,
    T,
    U,
):

    mu_x = float(np.asarray(dic_adim_norm["mu_x"]).squeeze())
    mu_y = float(np.asarray(dic_adim_norm["mu_y"]).squeeze())
    mu_z = float(np.asarray(dic_adim_norm["mu_z"]).squeeze())
    mu_t = float(np.asarray(dic_adim_norm["mu_t"]).squeeze())

    sigma_x = float(np.asarray(dic_adim_norm["sigma_x"]).squeeze())
    sigma_y = float(np.asarray(dic_adim_norm["sigma_y"]).squeeze())
    sigma_z = float(np.asarray(dic_adim_norm["sigma_z"]).squeeze())
    sigma_t = float(np.asarray(dic_adim_norm["sigma_t"]).squeeze())

    Nt, Np, _ = Data_acc_mks.shape

    Data_acc_flat = Data_acc_mks.reshape(-1, 7)
    Data_adv_flat = Data_adv_mks.reshape(-1, 7)
    Data_visc_flat = Data_visc_mks.reshape(-1, 7)

    coords = Data_acc_flat[:, 0:4]

    x = coords[:, 0:1]
    y = coords[:, 1:2]
    z = coords[:, 2:3]
    t = coords[:, 3:4]

    x_norm = (x / L - mu_x) / sigma_x
    y_norm = (y / L - mu_y) / sigma_y
    z_norm = (z / L - mu_z) / sigma_z
    t_norm = (t / T - mu_t) / sigma_t

    X_norm = np.concatenate(
        [x_norm, y_norm, z_norm, t_norm],
        axis=1,
    )

    scale = L / (U ** 2)

    grad_p_acc = -scale * Data_acc_flat[:, 4:7]
    grad_p_adv = -scale * Data_adv_flat[:, 4:7]
    grad_p_vis =  scale * Data_visc_flat[:, 4:7]

    Y_grad_pressure = np.concatenate(
        [
            grad_p_acc,
            grad_p_adv,
            grad_p_vis,
        ],
        axis=1,
    )

    return X_norm, Y_grad_pressure


# Predict pressure components on the centerline
def predict_pressure_on_centerline(
    data_grid,
    centerline_xyz,
    net_pres,
    L,
    T,
    mu,
    sigma,
    mmhg_scale,
):

    data_grid = np.asarray(data_grid)
    centerline_xyz = np.asarray(centerline_xyz)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if data_grid.ndim != 3 or data_grid.shape[2] < 4:
        raise ValueError("data_grid debe tener forma (Nt, Np, >=4).")
    if centerline_xyz.ndim != 2 or centerline_xyz.shape[1] < 3:
        raise ValueError("centerline_xyz debe tener forma (Np, >=3).")
    if mu.shape != (4,) or sigma.shape != (4,):
        raise ValueError("mu y sigma deben contener cuatro valores: x, y, z, tiempo.")
    if np.any(sigma == 0):
        raise ValueError("Los valores de sigma no pueden ser cero.")

    Nt = data_grid.shape[0]
    Np = centerline_xyz.shape[0]

    time_values = np.asarray(data_grid[:, 0, 3], dtype=np.float64)
    xyz = np.broadcast_to(centerline_xyz[None, :, :3], (Nt, Np, 3))
    time_grid = np.broadcast_to(time_values[:, None, None], (Nt, Np, 1))
    X = np.concatenate([xyz, time_grid], axis=2)

    X_nd = X.reshape(-1, 4).copy()
    X_nd[:, :3] /= L
    X_nd[:, 3] /= T
    X_norm = (X_nd - mu) / sigma

    model_parameter = next(net_pres.parameters(), None)
    if model_parameter is None:
        model_dtype = torch.float64
        model_device = torch.device("cpu")
    else:
        model_dtype = model_parameter.dtype
        model_device = model_parameter.device

    with torch.no_grad():
        X_tensor = torch.as_tensor(
            X_norm, dtype=model_dtype, device=model_device
        )
        components = net_pres(
            X_tensor[:, 0:1],
            X_tensor[:, 1:2],
            X_tensor[:, 2:3],
            X_tensor[:, 3:4],
        ) * mmhg_scale

    if components.ndim != 2 or components.shape[1] != 3:
        raise ValueError("net_pres must return three pressure components.")

    components_np = components.cpu().numpy().reshape(Nt, Np, 3)
    total_np = components_np.sum(axis=2, keepdims=True)

    return np.concatenate([X, components_np, total_np], axis=2)


# Predict pressure components on the tubular region
def predict_pressure_on_tubular_region(
    data_grid,
    net_pres,
    L,
    T,
    mu,
    sigma,
    mmhg_scale,
):
    import torch

    data_grid = np.asarray(data_grid)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    if data_grid.ndim != 3 or data_grid.shape[2] < 4:
        raise ValueError("data_grid must have shape (Nt, Np, >=4).")
    if mu.shape != (4,) or sigma.shape != (4,):
        raise ValueError("mu and sigma must contain four values: x, y, z, time.")
    if np.any(sigma == 0):
        raise ValueError("Sigma values cannot be zero.")

    Nt = data_grid.shape[0]
    Np = data_grid.shape[1]

    # Mantener las coordenadas espacio-temporales propias de cada punto.
    X = np.asarray(data_grid[:, :, :4], dtype=np.float64)

    X_nd = X.reshape(-1, 4).copy()
    X_nd[:, :3] /= L
    X_nd[:, 3] /= T
    X_norm = (X_nd - mu) / sigma

    model_parameter = next(net_pres.parameters(), None)
    if model_parameter is None:
        model_dtype = torch.float64
        model_device = torch.device("cpu")
    else:
        model_dtype = model_parameter.dtype
        model_device = model_parameter.device

    with torch.no_grad():
        X_tensor = torch.as_tensor(
            X_norm, dtype=model_dtype, device=model_device
        )
        components = net_pres(
            X_tensor[:, 0:1],
            X_tensor[:, 1:2],
            X_tensor[:, 2:3],
            X_tensor[:, 3:4],
        ) * mmhg_scale

    if components.ndim != 2 or components.shape[1] != 3:
        raise ValueError("net_pres must return three pressure components.")

    components_np = components.cpu().numpy().reshape(Nt, Np, 3)
    total_np = components_np.sum(axis=2, keepdims=True)

    return np.concatenate([X, components_np, total_np], axis=2)