from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch


#####################################################################
def plot_velocity_panels(
    Data_grid,
    time_index=0,
    view_angles=((45, 45), (20, 135)),
    vector_scale=0.02,
    normalize_vectors=False,
    max_vectors=5000,
    max_segmentation_points=50000,
    output_path=None,
):
    """
    Plot one time frame in three panels:
      1. Segmentation.
      2. Segmentation and velocity vectors.
      3. Same as panel 2, viewed from another angle.

    Data_grid: shape (Nt, Np, >=7), columns x, y, z, t, u, v, w.
    vector_scale: arrow length multiplier.
    normalize_vectors: use equal arrow lengths when True;
                       color still represents velocity magnitude.
    """
    Data_grid = np.asarray(Data_grid)

    frame = Data_grid[time_index]
    xyz = frame[:, :3]
    uvw = frame[:, 4:7]

    # Fixed samples shared across panels.
    rng = np.random.default_rng(42)

    segmentation_indices = np.arange(xyz.shape[0])
    segmentation_indices = rng.choice(
        segmentation_indices,
        size=min(max_segmentation_points, len(segmentation_indices)),
        replace=False,
    )

    velocity_indices = np.arange(uvw.shape[0])
    velocity_indices = rng.choice(
        velocity_indices,
        size=min(max_vectors, len(velocity_indices)),
        replace=False,
    )

    segmentation = xyz[segmentation_indices]
    vector_origins = xyz[velocity_indices]
    vectors = uvw[velocity_indices]

    # Shared color scale for the selected time.
    speed_all = np.linalg.norm(uvw, axis=1)
    speed = np.linalg.norm(vectors, axis=1)
    vmin, vmax = speed_all.min(), speed_all.max()

    if vmin == vmax:
        vmax = vmin + max(abs(vmin) * 0.01, 1e-12)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("jet")
    if normalize_vectors:
        vectors = vectors / (speed[:, np.newaxis] + 1e-12)
    colors = cmap(norm(speed))

    # Shared limits, with padding for flat or narrow geometries.
    lower = xyz.min(axis=0)
    upper = xyz.max(axis=0)
    span = upper - lower
    reference_span = max(span.max(), 1e-12)
    padding = np.maximum(0.03 * span, 0.01 * reference_span)
    lower -= padding
    upper += padding

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )

    titles = [
        "Segmentation",
        "Segmentation + velocity",
        "Segmentation + velocity — second view",
    ]
    panel_views = [view_angles[0], view_angles[0], view_angles[1]]

    for panel, (ax, title, (elev, azim)) in enumerate(
        zip(axes, titles, panel_views)
    ):
        ax.scatter(
            *segmentation.T,
            color="lightgray",
            s=2,
            alpha=0.25,
            linewidths=0,
        )

        if panel > 0:
            # Matplotlib draws all shafts first, then two head segments
            # per arrow. Match their colors to the corresponding vectors.
            quiver_colors = np.concatenate(
                [colors, np.repeat(colors, 2, axis=0)]
            )

            ax.quiver(
                *vector_origins.T,
                *vectors.T,
                length=vector_scale,
                normalize=normalize_vectors,
                colors=quiver_colors,
                linewidth=0.8,
                arrow_length_ratio=0.25,
            )

        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        ax.set_box_aspect(upper - lower)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title, fontsize=14)

    t_value = frame[0, 3]
    fig.suptitle(f"t = {t_value:.4f} s", fontsize=18)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes[1:].tolist(), shrink=0.7, pad=0.02)
    cbar.set_label(r"$\|\mathbf{u}\|$ [m/s]", fontsize=13)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path, dpi=300, bbox_inches="tight", facecolor="white"
        )
        print(f"Saved figure: {output_path}")

    return fig, axes


#####################################################################
def plot_pressure_panels(
    Data_grid,
    time_index=0,
    view_angles=((45, 45), (20, 135)),
    max_points=50000,
    point_size=3,
    cmap_name="cool_r",
    pressure_unit=None,
    output_path=None,
):
    """
    Plot one time frame in three panels:
      1. Segmentation.
      2. Pressure field.
      3. Pressure field from another angle.

    Data_grid: shape (Nt, Np, >=8), columns x, y, z, t, u, v, w, p.
    pressure_unit: label only; no unit conversion is performed.
    """
    Data_grid = np.asarray(Data_grid)

    frame = Data_grid[time_index]
    xyz = frame[:, :3]
    pressure = frame[:, 7]

    valid_xyz = np.isfinite(xyz).all(axis=1)
    valid_pressure = valid_xyz & np.isfinite(pressure)

    if not valid_pressure.any():
        raise ValueError("No valid pressure points in the selected frame.")

    # Spatial sample shared across all panels.
    rng = np.random.default_rng(42)
    indices = np.flatnonzero(valid_xyz)
    indices = rng.choice(
        indices,
        size=min(max_points, len(indices)),
        replace=False,
    )

    segmentation = xyz[indices]
    pressure_indices = indices[valid_pressure[indices]]

    if len(pressure_indices) == 0:
        raise ValueError("No valid pressure points in the sample; increase max_points.")

    pressure_xyz = xyz[pressure_indices]
    pressure_values = pressure[pressure_indices]

    # Color limits use all valid pressures in the selected frame.
    pmin = pressure[valid_pressure].min()
    pmax = pressure[valid_pressure].max()

    if pmin == pmax:
        delta = max(abs(pmin) * 0.01, 1e-12)
        pmin -= delta
        pmax += delta

    norm = Normalize(vmin=pmin, vmax=pmax)
    cmap = plt.get_cmap(cmap_name)

    # Shared spatial limits and proportions.
    lower = xyz[valid_xyz].min(axis=0)
    upper = xyz[valid_xyz].max(axis=0)
    span = upper - lower
    reference_span = max(span.max(), 1e-12)
    padding = np.maximum(0.03 * span, 0.01 * reference_span)
    lower -= padding
    upper += padding

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )

    titles = [
        "Segmentation",
        "Pressure",
        "Pressure — second view",
    ]
    panel_views = [view_angles[0], view_angles[0], view_angles[1]]

    for panel, (ax, title, (elev, azim)) in enumerate(
        zip(axes, titles, panel_views)
    ):
        if panel == 0:
            ax.scatter(
                *segmentation.T,
                color="lightgray",
                s=point_size,
                alpha=0.3,
                linewidths=0,
            )
        else:
            pressure_plot = ax.scatter(
                *pressure_xyz.T,
                c=pressure_values,
                cmap=cmap,
                norm=norm,
                s=point_size,
                alpha=1.0,
                linewidths=0,
                depthshade=False,
            )

        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        ax.set_box_aspect(upper - lower)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title, fontsize=14)

    t_value = frame[0, 3]
    fig.suptitle(f"t = {t_value:.4f} s", fontsize=18)

    cbar = fig.colorbar(
        pressure_plot,
        ax=axes[1:].tolist(),
        shrink=0.7,
        pad=0.02,
    )
    label = "Pressure"
    if pressure_unit:
        label += f" [{pressure_unit}]"
    cbar.set_label(label, fontsize=13)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"Saved figure: {output_path}")

    return fig, axes

#####################################################################
def plot_segmentation_with_centerline(
    Data_grid,
    centerline_xyz,
    skel_xyz=None,
    view_angle=(50, 25),
    max_segmentation_points=50000,
    show_skeleton=False,
    output_path=None,
):
    """Plot segmentation and centerline in one static panel.

    Data_grid: (Nt, Np, >=3), with x, y, z in the first columns.
    centerline_xyz, skel_xyz: physical coordinates with shape (N, 3).
    view_angle: (elevation, azimuth), in degrees.
    Returns fig, ax; call plt.show() to display the figure.
    """
    xyz = np.asarray(Data_grid)[0, :, :3]
    centerline_xyz = np.asarray(centerline_xyz)

    rng = np.random.default_rng(42)
    indices = rng.choice(
        len(xyz), min(max_segmentation_points, len(xyz)), replace=False
    )
    segmentation = xyz[indices]

    fig, ax = plt.subplots(
        figsize=(7, 7),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )

    ax.scatter(
        *segmentation.T, color="lightgray", s=2, alpha=0.25,
        linewidths=0, label="Segmentation",
    )

    if show_skeleton:
        if skel_xyz is None:
            raise ValueError("Provide skel_xyz when show_skeleton=True.")
        ax.scatter(
            *np.asarray(skel_xyz).T, color="green", s=8,
            alpha=0.35, label="Skeleton",
        )

    ax.plot(*centerline_xyz.T, color="blue", linewidth=3, label="Centerline")
    ax.scatter(*centerline_xyz[0], color="black", s=50, label="Start")
    ax.scatter(
        *centerline_xyz[-1], color="yellow", edgecolor="black",
        s=60, label="End",
    )

    # Include the centerline when setting the limits.
    lower = np.minimum(xyz.min(axis=0), centerline_xyz.min(axis=0))
    upper = np.maximum(xyz.max(axis=0), centerline_xyz.max(axis=0))
    span = upper - lower
    padding = np.maximum(0.03 * span, 0.01 * max(span.max(), 1e-12))
    lower -= padding
    upper += padding

    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(upper - lower)
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    ax.set_axis_off()
    ax.set_title("Segmentation + Centerline", fontsize=16)
    ax.legend()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

        print(f"Saved figure: {output_path}")

    return fig, ax



#####################################################################
def plot_tubular_region(
    coords_tube_ordered,
    Data_grid,
    radius,
    view_angle=(30, 25),
    output_path=None,
):

    x_ = Data_grid[0, :, 0]
    y_ = Data_grid[0, :, 1]
    z_ = Data_grid[0, :, 2]


    fig, ax = plt.subplots(
        figsize=(7, 7),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )

    ax.scatter(coords_tube_ordered[:, 0], coords_tube_ordered[:, 1], coords_tube_ordered[:, 2], color='red', s=1)   
    ax.scatter(x_, y_, z_, color='lightgray', alpha=0.1, s=0.1)
    ax.set_title(f'Tubular region with radius $r = {radius}$ m', fontsize=15)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_axis_off()
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

        print(f"Saved figure: {output_path}")

    return fig, ax



#####################################################################
def plot_centerline_velocity_panels(
    Data_grid,
    data_centerline,
    time_index=0,
    view_angles=((45, 45), (20, 135)),
    vector_scale=0.009,
    normalize_vectors=False,
    max_vectors=15000,
    max_segmentation_points=50000,
    output_path=None,
):
    Data_grid = np.asarray(Data_grid)
    data_centerline = np.asarray(data_centerline)

    geometry = Data_grid[0, :, :3]
    frame = data_centerline[time_index]
    xyz, uvw = frame[:, :3], frame[:, 4:7]

    rng = np.random.default_rng(42)
    idx_seg = rng.choice(
        len(geometry), min(max_segmentation_points, len(geometry)), replace=False
    )
    idx_vec = rng.choice(len(xyz), min(max_vectors, len(xyz)), replace=False)
    segmentation = geometry[idx_seg]
    vector_origins, vectors = xyz[idx_vec], uvw[idx_vec]
    speed_frame = np.linalg.norm(uvw, axis=-1)
    speed = speed_frame[idx_vec]

    speed_all = np.linalg.norm(data_centerline[:, :, 4:7], axis=-1)
    speed_all = speed_all[np.isfinite(speed_all)]
    if len(speed_all) == 0:
        raise ValueError("data_centerline contains no finite velocity magnitudes.")
    vmin, vmax = speed_all.min(), speed_all.max()
    if vmin == vmax:
        vmax = vmin + max(abs(vmin) * 0.01, 1e-12)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("jet")
    colors = cmap(norm(speed))
    # Quiver stores shafts first, then two head segments per arrow.
    quiver_colors = np.concatenate([colors, np.repeat(colors, 2, axis=0)])

    lower = np.nanmin(Data_grid[:, :, :3], axis=(0, 1))
    upper = np.nanmax(Data_grid[:, :, :3], axis=(0, 1))
    span = upper - lower
    padding = np.maximum(0.03 * span, 0.01 * max(span.max(), 1e-12))
    lower -= padding
    upper += padding

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6),
        subplot_kw={"projection": "3d"}, constrained_layout=True,
    )

    for panel, (ax, (elev, azim)) in enumerate(zip(axes, view_angles), start=1):
        ax.scatter(
            *segmentation.T, color="lightgray", s=2, alpha=0.3, linewidths=0
        )
        if len(vectors):
            ax.quiver(
                *vector_origins.T, *vectors.T,
                length=vector_scale, normalize=normalize_vectors,
                colors=quiver_colors, linewidth=0.6, arrow_length_ratio=0.25,
            )
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        ax.set_box_aspect(upper - lower)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f"Centerline-region velocity | view {panel}", fontsize=14)

    fig.suptitle(f"t = {frame[0, 3]:.4f} s", fontsize=18)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.7, pad=0.02)
    cbar.set_label(r"$\|\mathbf{u}\|$ [m/s]", fontsize=13)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")

        print(f"Saved figure: {output_path}")

    return fig, axes


#####################################################################
def plot_pressure_drop_panels(
    segmentation_xyz,
    spatial_points,
    predicted_total_pressure,
    centerline_xyz,
    p1_xyz,
    p2_xyz,
    time_value,
    time_values,
    drop_components,
    drop_total,
    drop_cfd,
    p1_index,
    p2_index,
    cfd_p1_index=None,
    cfd_p2_index=None,
    figure_size=(20, 7),
    panel_widths=(2.0, 1.0),
    view_angle=(45, 45),
    max_segmentation_points=50000,
    output_path=None,
):
    """Plot predicted pressure in the segmented region and P1-P2 pressure drops.

    Spatial arrays contain xyz coordinates in physical units. Pressure arrays
    are expected in mmHg. ``drop_components`` maps component names to arrays.
    Returns the figure and the spatial and pressure-drop axes.
    """
    segmentation_xyz = np.asarray(segmentation_xyz)
    spatial_points = np.asarray(spatial_points)
    predicted_total_pressure = np.asarray(predicted_total_pressure)
    centerline_xyz = np.asarray(centerline_xyz)
    p1_xyz = np.asarray(p1_xyz)
    p2_xyz = np.asarray(p2_xyz)
    time_values = np.asarray(time_values)

    if segmentation_xyz.ndim != 2 or segmentation_xyz.shape[1] < 3:
        raise ValueError("segmentation_xyz must have shape (N, >=3).")
    if spatial_points.ndim != 2 or spatial_points.shape[1] < 3:
        raise ValueError("spatial_points must have shape (N, >=3).")
    if predicted_total_pressure.shape[0] != spatial_points.shape[0]:
        raise ValueError("There must be a predicted pressure for each spatial point.")
    if centerline_xyz.ndim != 2 or centerline_xyz.shape[1] < 3:
        raise ValueError("centerline_xyz must have shape (N, >=3).")
    if p1_xyz.shape[0] < 3 or p2_xyz.shape[0] < 3:
        raise ValueError("p1_xyz and p2_xyz must contain three coordinates.")
    if len(panel_widths) != 2 or any(width <= 0 for width in panel_widths):
        raise ValueError("panel_widths must contain two positive values.")

    if len(segmentation_xyz) > max_segmentation_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(
            len(segmentation_xyz), max_segmentation_points, replace=False
        )
        segmentation_xyz = segmentation_xyz[indices]

    fig = plt.figure(figsize=figure_size)
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(1, 2, width_ratios=panel_widths)
    ax_spatial = fig.add_subplot(grid[0, 0], projection="3d")
    ax_drop = fig.add_subplot(grid[0, 1])
    ax_spatial.set_facecolor("white")

    for axis in (ax_spatial.xaxis, ax_spatial.yaxis, ax_spatial.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("white")

    ax_spatial.scatter(
        *segmentation_xyz[:, :3].T,
        color="grey",
        s=1,
        alpha=0.11,
        linewidths=0,
        depthshade=False,
    )
    pressure_scatter = ax_spatial.scatter(
        *spatial_points[:, :3].T,
        c=predicted_total_pressure,
        cmap="viridis",
        s=7,
        alpha=0.8,
        linewidths=0,
        # label="Tubular region: pressure prediction",
        depthshade=False,
    )
    ax_spatial.plot(
        *centerline_xyz[:, :3].T,
        color="black",
        linewidth=1.5,
        label="Centerline",
        zorder=10,
    )
    ax_spatial.scatter(
        *p1_xyz[:3],
        color="black",
        edgecolor="grey",
        s=300,
        marker="x",
        depthshade=False,
        label=f"$\pi_1$ (index {p1_index})",
        zorder=30,
    )
    ax_spatial.scatter(
        *p2_xyz[:3],
        color="black",
        edgecolor="grey",
        s=300,
        marker="x",
        depthshade=False,
        label=f"$\pi_2$ (index {p2_index})",
        zorder=30,
    )
    ax_spatial.text(
        *p1_xyz[:3], " $\pi_1$", color="black", fontsize=15, fontweight="bold"
    )
    ax_spatial.text(
        *p2_xyz[:3], " $\pi_2$", color="black", fontsize=15, fontweight="bold"
    )
    ax_spatial.set_title(
        f"Predicted total pressure, t = {time_value:.4f} s", fontsize=20
    )
    ax_spatial.view_init(elev=view_angle[0], azim=view_angle[1])
    ax_spatial.set_axis_off()

    colorbar = fig.colorbar(
        pressure_scatter, ax=ax_spatial, shrink=0.72, pad=0.1
    )
    colorbar.set_label("Pressure field estimation [mmHg]")

    component_styles = {
        "acc": ("green", "--"),
        "adv": ("blue", "--"),
        "vis": ("purple", "--"),
    }
    for component, values in drop_components.items():
        color, linestyle = component_styles.get(component, (None, "--"))
        ax_drop.plot(
            time_values,
            values,
            linewidth=2,
            color=color,
            linestyle=linestyle,
            label=rf"$\Delta p_{{\mathrm{{{component}}}}}^{{\mathrm{{PINN}}}}$",
        )

    ax_drop.plot(
        time_values,
        drop_total,
        linewidth=4,
        color="red",
        label=r"$\Delta p_{\mathrm{total}}^{\mathrm{PINN}}$",
    )
    ax_drop.plot(
        time_values,
        drop_cfd,
        linewidth=4,
        color="grey",
        linestyle="--",
        label=r"$\Delta p_{\mathrm{total}}^{\mathrm{CFD}}$",
    )
    ax_drop.set_xlabel("Time [s]", fontsize=18)
    ax_drop.set_ylabel(r"$\Delta p$ [mmHg]", fontsize=18)
    ax_drop.set_title(r"Predicted pressure drop $p(\pi_1,t)-p(\pi_2,t)$", fontsize=20)
    ax_drop.tick_params(labelsize=17)
    ax_drop.grid(True)
    ax_drop.legend(fontsize=15)

    fig.tight_layout(rect=(0, 0.16, 1, 1))
    spatial_handles, spatial_labels = ax_spatial.get_legend_handles_labels()
    segmentation_handle = Patch(
        facecolor="grey", edgecolor="grey", label="Full segmentation"
    )
    spatial_handles.insert(0, segmentation_handle)
    spatial_labels.insert(0, "Full segmentation")
    spatial_position = ax_spatial.get_position()
    fig.legend(
        spatial_handles,
        spatial_labels,
        loc="lower center",
        bbox_to_anchor=(
            spatial_position.x0 + spatial_position.width / 2,
            0.01,
        ),
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=12,
    )

    if output_path is not None:     
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved figure: {output_path}")

    return fig, (ax_spatial, ax_drop)