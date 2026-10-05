
# PINN-Based Pressure Drop Estimation and Decomposition
Two-stage physics-informed neural networks for non-invasive aortic pressure-drop estimation and decomposition into acceleration, advection, and viscous contributions. Includes in silico CFD example.

<p align="center">
  <a href="docs/Graphical_Abstract.pdf">
    <img
      src="docs/Graphical_Abstract.jpg"
      alt="Pipeline for aortic pressure-drop estimation and decomposition"
      width="1000"
    >
  </a>
</p>

**Figure 1. Overview of the proposed pipeline for aortic pressure-drop estimation and decomposition.** The green panel on the left (“Selection of 4D Flow MRI velocity data”) illustrates the selection of velocity measurements within a centerline-based tubular region, $\Omega_{\mathrm{CR}}$, with the selected spatial locations retained across all available time frames to construct the training dataset. In Stage 1, the selected data (a) train a velocity network (b) using a data-fidelity term and an incompressibility constraint. Automatic differentiation of the reconstructed velocity field provides the accelerative, advective, and viscous momentum contributions (c). In Stage 2, a pressure-component network (d) approximates the gradient-field projection of each momentum contribution by minimizing the mismatch between that contribution and the corresponding pressure-component gradient. The resulting scalar fields are combined into a relative pressure field (e), from which the pressure drop between the proximal and distal locations, $\pi_1$ and $\pi_2$, is evaluated and decomposed into its three contributions (f). The right panel illustrates validation across *in silico*, *in vitro*, and *in vivo* testbeds.
