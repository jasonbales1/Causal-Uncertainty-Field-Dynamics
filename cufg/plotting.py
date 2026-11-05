"""
Visualization Engine for the CUFG Benchmark Suite
=================================================
Generates publication-grade diagnostic plots for all 5 physical benchmarks.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "#0f141c",
        "axes.facecolor": "#161d28",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#e2e8f0",
        "axes.titlecolor": "#f8fafc",
        "xtick.color": "#cbd5e1",
        "ytick.color": "#cbd5e1",
        "text.color": "#e2e8f0",
        "grid.color": "#263346",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.facecolor": "#1e293b",
        "legend.edgecolor": "#475569",
        "savefig.facecolor": "#0f141c",
        "savefig.dpi": 180,
    })


def generate_all_plots(results: dict, out_dir: Path) -> list[Path]:
    """Generates all benchmark figures and returns their file paths."""
    setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    # -------------------------------------------------------------------------
    # Figure 1: Solar System 1/r^2 Precision & Causal c-Propagation Orbit Test
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    ss = results["solar_system"]
    orb = results["retarded_orbit"]

    r_au = ss["r_au_grid"]
    ax1.plot(r_au, ss["family_curves"]["rar"]["log10_frac_dev"], color="#38bdf8", lw=2.4,
             label="CUFG Quantum Bose-Einstein Overlap: 1/(1 - exp(-√y))")
    ax1.plot(r_au, ss["family_curves"]["standard"]["log10_frac_dev"], color="#a78bfa", lw=1.8, ls="--",
             label="Quadrupole Algebraic Overlap (p=2)")
    ax1.plot(r_au, ss["family_curves"]["simple"]["log10_frac_dev"], color="#f87171", lw=1.8, ls=":",
             label="Naive Dipole Overlap (p=1, ruled out)")

    # Plot planetary observational bounds with custom staggered offsets so zero labels overlap
    planet_offsets = {
        "Mercury": (-14, -18),
        "Venus": (-12, -25),
        "Earth": (0, -23),
        "Mars": (2, -14),
        "Jupiter": (-6, -24),
        "Saturn": (8, -18),
        "Uranus": (-16, -20),
        "Neptune": (12, -8),
    }
    first_bound = True
    for pl in ss["planet_results"]:
        if pl["bound"] is not None:
            short_name = pl["name"].split(" ")[0]
            dx, dy = planet_offsets.get(short_name, (4, 8))
            lbl_scatter = "Observational Upper Bounds (▼)" if first_bound else None
            first_bound = False
            ax1.scatter([pl["r_au"]], [np.log10(pl["bound"])], color="#facc15", s=65, zorder=5,
                        marker="v", label=lbl_scatter)
            ax1.annotate(short_name, (pl["r_au"], np.log10(pl["bound"])),
                         xytext=(dx, dy), textcoords="offset points", fontsize=8.0, color="#facc15",
                         arrowprops=dict(arrowstyle="-", color="#facc15", lw=0.7, alpha=0.65))

    ax1.set_xscale("log")
    ax1.set_ylim(-45, 0.5)
    ax1.set_xlim(0.3, 1e5)
    ax1.set_xlabel("Heliocentric Distance r [AU]")
    ax1.set_ylabel(r"$\log_{10}(\Delta g / g_{\mathrm{pair}})$")
    ax1.set_title("Solar System 1/r² Precision vs. Overlap Screening")
    ax1.grid(True)
    ax1.legend(loc="lower right", fontsize=8.2)

    # Panel B: Retarded c-propagation orbital stability + Mercury Precession
    ax2.plot(orb["orbit_num"], orb["r_naive_retarded"], color="#f87171", lw=2.0, ls="--",
             label="Naive Position Retardation r(t - R/c) (Unstable)")
    ax2.plot(orb["orbit_num"], orb["r_covariant_retarded"], color="#34d399", lw=2.5,
             label="Covariant Phase-Space Uncertainty at c (Stable)")
    ax2.set_xlabel("Number of Orbits")
    ax2.set_ylabel("Normalized Orbital Radius r(t) / r₀")
    ax2.set_title("Causal Propagation at Speed c: Orbital Stability Test")
    ax2.grid(True)
    ax2.legend(loc="upper left", fontsize=8.5)

    # Add inset text box with Mercury Precession & Light Deflection results
    info_txt = (
        f"Relativistic PPN Benchmarks (c-propagation):\n"
        f"• Mercury Precession:\n"
        f"  - Scalar ΔtΔE only (γ=0): {orb['precession_scalar_uncertainty']:.2f}\" / century\n"
        f"  - 4D Phase-Space (γ=1):   {orb['precession_4d_phasespace_uncertainty']:.2f}\" / century\n"
        f"  - Observed:               {orb['mercury_observed_arcsec_century']:.2f} ± {orb['mercury_observed_err']:.2f}\"\n"
        f"• Solar Light Bending:\n"
        f"  - 4D Phase-Space (γ=1):   {orb['light_deflection_4d_phasespace_arcsec']:.3f}\" (Obs: 1.751\")"
    )
    ax2.text(0.04, 0.42, info_txt, transform=ax2.transAxes, fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#38bdf8", alpha=0.9))

    fig.tight_layout()
    p1 = out_dir / "fig1_solar_system_and_causal_c.png"
    fig.savefig(p1)
    plt.close(fig)
    saved_paths.append(p1)

    # -------------------------------------------------------------------------
    # Figure 2: 2D FDTD Wave Propagation at c & Multi-Body Overlap / Void Map
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    wave = results["wave_2d"]
    mbody = results["multibody_2d"]

    # Panel A: Wavefront propagation at c
    colors_snap = ["#38bdf8", "#a78bfa", "#34d399"]
    for idx, (step, snap) in enumerate(wave["snapshots"].items()):
        axes[0].plot(snap["r_axis"], snap["psi_radial"], color=colors_snap[idx], lw=2.0,
                     label=f"t = {snap['time']:.0f} Δx/c")
        axes[0].axvline(snap["time"], color=colors_snap[idx], ls=":", alpha=0.6)
    axes[0].set_xlim(0, 70)
    axes[0].set_xlabel("Radial Distance r [c·Δt units]")
    axes[0].set_ylabel("Uncertainty Wave Amplitude ψ(r, t)")
    axes[0].set_title(f"2D Causal Wavefront Propagation (v_measured = {wave['measured_v_over_c']:.3f} c)")
    axes[0].grid(True)
    axes[0].legend(fontsize=8.5)

    # Panel B: Multi-Body Anomalous Overlap Acceleration Halo & Bridge
    coords = mbody["coords_kpc"]
    im1 = axes[1].imshow(
        mbody["anomalous_accel_a0"],
        extent=[coords[0], coords[-1], coords[0], coords[-1]],
        origin="lower",
        cmap="magma",
        vmin=0.04,
        vmax=0.65,
    )
    axes[1].contour(
        coords,
        coords,
        mbody["anomalous_accel_a0"],
        levels=[0.10, 0.20, 0.38],
        colors=["#38bdf8", "#facc15", "#ffffff"],
        linewidths=0.9,
        alpha=0.75,
    )
    for src in mbody["sources"]:
        axes[1].scatter([src["x_kpc"]], [src["y_kpc"]], color="#38bdf8", edgecolor="white", s=55, zorder=5)
    axes[1].set_xlabel("X [kpc]")
    axes[1].set_ylabel("Y [kpc]")
    axes[1].set_title("Overlap Amplification Halo (|g_CUFG| - |g_pair|) / a₀")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Extra Overlap Accel [a₀]")

    # Panel C: Local Empty-Space Expansion Map H_local / H_void
    im2 = axes[2].imshow(
        mbody["local_expansion_ratio"],
        extent=[coords[0], coords[-1], coords[0], coords[-1]],
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    for src in mbody["sources"]:
        axes[2].scatter([src["x_kpc"]], [src["y_kpc"]], color="#f87171", edgecolor="white", s=55, zorder=5)
    axes[2].set_xlabel("X [kpc]")
    axes[2].set_ylabel("Y [kpc]")
    axes[2].set_title("Local Space Expansion Rate H_local(x) / H_void")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="S_void (0=Bound, 1=Expanding)")

    fig.tight_layout()
    p2 = out_dir / "fig2_wave_overlap_and_void_map.png"
    fig.savefig(p2)
    plt.close(fig)
    saved_paths.append(p2)

    # -------------------------------------------------------------------------
    # Figure 3: Galactic Rotation Curves (NGC 3198 & DDO 154) + BTFR
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    gal = results["galaxies"]

    # Panel A: NGC 3198
    ngc = gal["ngc3198"]
    axes[0].errorbar(ngc["obs"][:, 0], ngc["obs"][:, 1], yerr=ngc["obs"][:, 2],
                     fmt="o", color="#facc15", markersize=5, label="SPARC Observed (NGC 3198)")
    axes[0].plot(ngc["r_kpc"], ngc["v_cufg_kms"], color="#38bdf8", lw=2.6,
                 label=f"CUFG Overlap (Zero DM, χ²={ngc['chi2_cufg']:.2f})")
    axes[0].plot(ngc["r_kpc"], ngc["v_bar_kms"], color="#f87171", lw=2.0, ls="--",
                 label=f"Unamplified Baryons g_pair (χ²={ngc['chi2_newton']:.1f})")
    axes[0].plot(ngc["r_kpc"], ngc["v_star_kms"], color="#94a3b8", lw=1.2, ls=":", label="Stellar Disk")
    axes[0].plot(ngc["r_kpc"], ngc["v_gas_kms"], color="#64748b", lw=1.2, ls="-.", label="HI Gas Disk")
    axes[0].set_xlabel("Galactocentric Radius r [kpc]")
    axes[0].set_ylabel("Circular Velocity V(r) [km/s]")
    axes[0].set_title("Spiral Galaxy NGC 3198 Rotation Curve")
    axes[0].set_ylim(0, 180)
    axes[0].grid(True)
    axes[0].legend(loc="center right", bbox_to_anchor=(0.98, 0.58), fontsize=7.8)

    # Panel B: DDO 154 (Low-Surface-Brightness Dwarf)
    ddo = gal["ddo154"]
    axes[1].errorbar(ddo["obs"][:, 0], ddo["obs"][:, 1], yerr=ddo["obs"][:, 2],
                     fmt="o", color="#facc15", markersize=5, label="SPARC Observed (DDO 154)")
    axes[1].plot(ddo["r_kpc"], ddo["v_cufg_kms"], color="#34d399", lw=2.6,
                 label=f"CUFG Overlap (Zero DM, χ²={ddo['chi2_cufg']:.2f})")
    axes[1].plot(ddo["r_kpc"], ddo["v_bar_kms"], color="#f87171", lw=2.0, ls="--",
                 label=f"Unamplified Baryons g_pair (χ²={ddo['chi2_newton']:.1f})")
    axes[1].set_xlabel("Galactocentric Radius r [kpc]")
    axes[1].set_ylabel("Circular Velocity V(r) [km/s]")
    axes[1].set_title("Gas-Rich Dwarf Galaxy DDO 154 (Deep Overlap Regime)")
    axes[1].set_ylim(0, 60)
    axes[1].grid(True)
    axes[1].legend(loc="center right", bbox_to_anchor=(0.98, 0.45), fontsize=7.8)

    # Panel C: Baryonic Tully-Fisher Relation
    btfr = gal["btfr"]
    axes[2].plot(btfr["v_flat_cufg_kms"], btfr["mass_msun"], color="#38bdf8", lw=2.6,
                 label=r"CUFG Prediction: $M_b = (M_{\mathrm{UV}} / \ell_{\mathrm{UV}} c^2 a_{0,\mathrm{eff}}) V_{\mathrm{flat}}^4$")
    axes[2].plot(btfr["v_out_newton_kms"], btfr["mass_msun"], color="#f87171", lw=2.0, ls="--",
                 label="Unamplified Pairwise Virial Scaling (No DM)")
    # Plot NGC 3198 and DDO 154 on the BTFR diagram
    axes[2].scatter([150.0, 47.0], [4.3e10, 3.4e8], color="#facc15", s=70, zorder=5,
                    label="NGC 3198 & DDO 154 Benchmarks")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Asymptotic Flat Rotation Speed V_flat [km/s]")
    axes[2].set_ylabel(r"Total Baryonic Mass $M_b$ [$M_\odot$]")
    axes[2].set_title("Baryonic Tully-Fisher Relation (Slope = 4)")
    axes[2].set_xlim(3.0, 350.0)
    axes[2].set_ylim(5e6, 9e11)
    axes[2].grid(True)
    axes[2].legend(loc="upper left", fontsize=8.0)

    fig.tight_layout()
    p3 = out_dir / "fig3_galaxy_rotation_and_btfr.png"
    fig.savefig(p3)
    plt.close(fig)
    saved_paths.append(p3)

    # -------------------------------------------------------------------------
    # Figure 4: Empty-Space Driven Expansion H(z), Void Fraction & q(z)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    cosmo = results["cosmology"]
    z = cosmo["z_grid"]

    # Panel A: H(z) vs Cosmic Chronometers & BAO
    obs_hz = cosmo["obs_hz"]
    axes[0].errorbar(obs_hz[:, 0], obs_hz[:, 1], yerr=obs_hz[:, 2],
                     fmt="o", color="#facc15", markersize=4.5, alpha=0.85, label="Cosmic Chronometers & BAO")
    axes[0].plot(z, cosmo["H_cufg"], color="#38bdf8", lw=2.5,
                 label=f"CUFG Empty-Space Expansion (H₀={cosmo['H0_cufg_today']:.1f}, χ²={cosmo['chi2_hz_cufg']:.2f})")
    axes[0].plot(z, cosmo["H_lcdm"], color="#a78bfa", lw=2.0, ls="--",
                 label=f"Standard ΛCDM (H₀={cosmo['H0_lcdm_today']:.1f}, χ²={cosmo['chi2_hz_lcdm']:.2f})")
    axes[0].plot(z, cosmo["H_no_void_exp"], color="#f87171", lw=1.6, ls=":",
                 label=f"No Empty-Space Expansion (χ²={cosmo['chi2_hz_novoid']:.1f})")
    axes[0].set_xlabel("Redshift z")
    axes[0].set_ylabel("Hubble Parameter H(z) [km/s/Mpc]")
    axes[0].set_title("Cosmic Expansion Rate H(z) vs. Observations")
    axes[0].grid(True)
    axes[0].legend(loc="upper left", fontsize=7.8)

    # Panel B: Void Volume Fraction f_void(z) & Deceleration q(z)
    axes[1].plot(z, cosmo["f_void_z"], color="#34d399", lw=2.5, label="Empty-Space Void Volume Fraction f_void(z)")
    axes[1].plot(z, cosmo["q_cufg"], color="#38bdf8", lw=2.2, label="CUFG Deceleration Parameter q(z)")
    axes[1].plot(z, cosmo["q_lcdm"], color="#a78bfa", lw=1.8, ls="--", label="ΛCDM Deceleration Parameter q(z)")
    axes[1].axhline(0.0, color="#94a3b8", lw=1.0, ls=":")
    axes[1].axvline(cosmo["z_acc_cufg"], color="#facc15", lw=1.2, ls="-.",
                    label=f"Acceleration Onset z_acc = {cosmo['z_acc_cufg']:.2f}")
    axes[1].set_xlabel("Redshift z")
    axes[1].set_ylabel("Dimensionless Parameter")
    axes[1].set_title("Void Growth & Onset of Cosmic Acceleration")
    axes[1].grid(True)
    axes[1].legend(loc="lower right", fontsize=8.0)

    # Panel C: Type Ia Supernova Distance Modulus Residual Delta mu(z)
    axes[2].plot(z[1:], cosmo["mu_cufg"][1:] - cosmo["mu_novoid"][1:], color="#38bdf8", lw=2.5,
                 label="CUFG Empty-Space Expansion - Open Matter")
    axes[2].plot(z[1:], cosmo["mu_lcdm"][1:] - cosmo["mu_novoid"][1:], color="#a78bfa", lw=2.0, ls="--",
                 label="Standard ΛCDM - Open Matter")
    axes[2].axhline(0.0, color="#f87171", lw=1.5, ls=":", label="Non-Accelerating Baseline")
    axes[2].set_xlabel("Redshift z")
    axes[2].set_ylabel("Distance Modulus Excess Δμ(z) [mag]")
    axes[2].set_title("Supernova Ia Acceleration Signature Δμ(z)")
    axes[2].grid(True)
    axes[2].legend(loc="lower right", fontsize=8.2)

    fig.tight_layout()
    p4 = out_dir / "fig4_cosmology_empty_space_expansion.png"
    fig.savefig(p4)
    plt.close(fig)
    saved_paths.append(p4)

    # -------------------------------------------------------------------------
    # Figure 5: Novel Falsifiable Predictions (Wide Binary EFE & Quantum Cat State)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
    wb = results["wide_binaries"]
    qlab = results["quantum_lab"]

    ax1.plot(wb["s_au"], wb["v_ratio_isolated"], color="#38bdf8", lw=2.5,
             label="Isolated Binary in Cosmic Void (g_ext = 0)")
    ax1.plot(wb["s_au"], wb["v_ratio_milky_way_efe"], color="#34d399", lw=2.5,
             label="Solar Neighborhood Binary (Milky Way EFE: g_ext = 1.75 a₀)")
    ax1.axhline(1.0, color="#f87171", lw=1.8, ls="--", label="Unamplified Pairwise Baseline (No Boost)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Binary Separation s [AU]")
    ax1.set_ylabel("Orbital Velocity Ratio V_eff / V_pair")
    ax1.set_title("Falsifiable Test 1: Wide Binary External Field Effect (EFE)")
    ax1.grid(True)
    ax1.legend(loc="upper left", fontsize=8.5)

    ax2.plot(qlab["d_um"], qlab["entanglement_rate_boost"], color="#facc15", lw=2.5,
             label="CUFG Self-Overlap Entanglement Boost (10 fg Nanocrystal)")
    ax2.axvline(qlab["quantum_transition_radius_um"], color="#38bdf8", lw=1.8, ls="-.",
                label=f"Quantum Transition Radius r_Q = {qlab['quantum_transition_radius_um']:.2f} μm")
    ax2.axhline(1.0, color="#f87171", lw=1.8, ls="--", label="Unamplified Pairwise Overlap Baseline")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Superposition Branch Separation d [μm]")
    ax2.set_ylabel("Entanglement Phase Rate Ratio Γ_CUFG / Γ_pair")
    ax2.set_title("Falsifiable Test 2: Deep-Space Optomechanical Cat-State Overlap")
    ax2.set_ylim(0.5, 3500.0)
    ax2.grid(True)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.11, 0.98), fontsize=8.3)

    fig.tight_layout()
    p5 = out_dir / "fig5_falsifiable_predictions.png"
    fig.savefig(p5)
    plt.close(fig)
    saved_paths.append(p5)

    # -------------------------------------------------------------------------
    # Figure 6: Bullet Cluster (1E 0657-56) Lensing Map & Collision Axis Profile
    # -------------------------------------------------------------------------
    if "bullet_cluster" in results:
        bc = results["bullet_cluster"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4))

        x_kpc = bc["x_kpc"]
        y_kpc = bc["y_kpc"]

        # Panel A: Iconic 2D Bullet Cluster Map:
        # Color heatmap = Shocked X-ray Intracluster Gas (Chandra analog)
        # Green/Cyan contours = Full Hyperbolic CUFG Weak-Lensing Convergence kappa(x, y)
        im_gas = ax1.imshow(
            bc["sigma_gas_2d"],
            extent=[x_kpc[0], x_kpc[-1], y_kpc[0], y_kpc[-1]],
            origin="lower",
            cmap="inferno",
            vmin=0.0,
            vmax=0.095,
            aspect="auto",
        )
        cs = ax1.contour(
            x_kpc,
            y_kpc,
            bc["kappa_hyperbolic_2d"],
            levels=[0.08, 0.13, 0.19, 0.25, 0.31],
            colors=["#34d399", "#34d399", "#38bdf8", "#38bdf8", "#ffffff"],
            linewidths=[1.2, 1.5, 1.8, 2.1, 2.3],
        )
        manual_locs = [(-250, 125), (355, -100), (355, 80), (-355, 95), (-355, -75), (-355, 58)]
        ax1.clabel(cs, inline=True, fontsize=7.5, fmt="κ=%.2f", manual=manual_locs)

        # Mark X-ray gas peaks and galaxy / lensing centroids
        ax1.scatter([-135.0, +115.0], [0.0, 0.0], color="#f87171", edgecolor="white",
                    s=85, marker="X", zorder=6, label="Shocked X-ray Gas Peaks (82% Baryons)")
        ax1.scatter([-355.0, +355.0], [0.0, 0.0], color="#38bdf8", edgecolor="white",
                    s=85, marker="*", zorder=6, label="Collisionless Galaxies & CUFG Lensing Peaks")

        ax1.set_xlabel("Collision Axis X [kpc]")
        ax1.set_ylabel("Transverse Axis Y [kpc]")
        ax1.set_title("Bullet Cluster (1E 0657-56): X-Ray Gas (Heatmap) vs. CUFG Lensing κ (Contours)")
        ax1.legend(loc="upper right", fontsize=8.0)
        fig.colorbar(im_gas, ax=ax1, fraction=0.046, pad=0.04, label="Normalized X-Ray Gas Density")

        # Panel B: 1D Collision-Axis Lensing Convergence Profile kappa(x, y=0)
        obs_b = bc["obs_bullet"]
        ax2.errorbar(obs_b[:, 0], obs_b[:, 1], yerr=obs_b[:, 2], fmt="o", color="#facc15",
                     markersize=5.5, zorder=6, label="Observed Lensing κ (Clowe et al. / Bradač et al.)")
        ax2.plot(x_kpc, bc["kappa_1d_hyperbolic"], color="#38bdf8", lw=2.7,
                 label=f"Full Hyperbolic CUFG (Wave Inertia + Multi-Galaxy Overlap, χ²={bc['chi2_hyperbolic']:.2f})")
        ax2.plot(x_kpc, bc["kappa_1d_static"], color="#a78bfa", lw=2.0, ls="--",
                 label=f"Static 1-Scale Overlap (Stuck on X-ray Gas, χ²={bc['chi2_static']:.1f})")
        ax2.plot(x_kpc, bc["kappa_1d_newton"], color="#f87171", lw=1.8, ls=":",
                 label=f"Unboosted Baryon Uncertainty (ν=1, Zero G, χ²={bc['chi2_newton']:.1f})")

        # Vertical markers for Gas vs Galaxy centroids
        ax2.axvline(-135.0, color="#f87171", alpha=0.45, ls="-.", lw=1.2)
        ax2.axvline(+115.0, color="#f87171", alpha=0.45, ls="-.", lw=1.2)
        ax2.axvline(-355.0, color="#34d399", alpha=0.45, ls="-.", lw=1.2)
        ax2.axvline(+355.0, color="#34d399", alpha=0.45, ls="-.", lw=1.2)

        ax2.annotate("", xy=(-350.0, 0.265), xytext=(-135.0, 0.265),
                     arrowprops=dict(arrowstyle="<->", color="#34d399", lw=1.8))
        ax2.text(-242.0, 0.275, f"Lensing–Gas Offset\nΔx = {bc['offset_hyperbolic_kpc']:.0f} kpc (Obs: 220 kpc)",
                 ha="center", va="bottom", fontsize=8.3, color="#34d399",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f172a", edgecolor="#34d399", alpha=0.85))

        ax2.set_xlim(-650, 650)
        ax2.set_ylim(0, 0.54)
        ax2.set_xlabel("Collision Axis X [kpc]")
        ax2.set_ylabel("Weak-Lensing Convergence κ(x, y=0)")
        ax2.set_title("1D Lensing Convergence Along Bullet Cluster Collision Axis")
        ax2.grid(True)
        ax2.legend(loc="upper right", fontsize=7.8)

        fig.tight_layout()
        p6 = out_dir / "fig6_bullet_cluster_lensing.png"
        fig.savefig(p6)
        plt.close(fig)
        saved_paths.append(p6)

    # -------------------------------------------------------------------------
    # Figure 7: UV Planck Limit, Proton-Neutron Bounce & Singularity Resolution
    # -------------------------------------------------------------------------
    if "uv_planck" in results:
        uv = results["uv_planck"]
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))

        # Panel A: Proton-Neutron Potential with 6-Quark / 9-Cross-Pair Multi-Quark Repulsive Core
        r_fm = uv["r_fm"]
        axes[0].plot(r_fm, uv["v_pn_total_mev"], color="#38bdf8", lw=2.6,
                     label="Total 9-Pair Quark-Overlap Potential V_eff(r)")
        axes[0].plot(r_fm, uv["v_heisenberg_rep_mev"], color="#f87171", lw=1.8, ls="--",
                     label="4/9 Like-Flavor (uu,dd) Pauli + Color-Mag Repulsion")
        axes[0].plot(r_fm, uv["v_overlap_att_mev"], color="#34d399", lw=1.8, ls=":",
                     label="5/9 Unlike-Flavor (ud) Chiral + 1π Attraction")
        axes[0].axhline(0.0, color="#94a3b8", lw=1.0)
        axes[0].axvline(uv["r_bounce_wall_fm"], color="#facc15", lw=1.5, ls="-.",
                        label=f"Multi-Quark Bounce Wall r = {uv['r_bounce_wall_fm']:.2f} fm")
        axes[0].set_xlim(0.25, 2.5)
        axes[0].set_ylim(-140, 450)
        axes[0].set_xlabel("Nucleon Separation r [fm]")
        axes[0].set_ylabel("Interaction Potential V(r) [MeV]")
        axes[0].set_title("Proton–Neutron 9-Pair Quark-Overlap Potential & Bounce Core")
        axes[0].grid(True)
        axes[0].legend(loc="upper right", fontsize=7.6)

        # Panel B: Head-On Proton-Neutron Collision Bounce Trajectories r(t)
        colors_tr = {50: "#34d399", 150: "#38bdf8", 400: "#facc15"}
        for e_cm, tr in uv["trajectories"].items():
            mask_tr = tr["r_fm"] <= 2.55
            axes[1].plot(tr["t_ys"][mask_tr], tr["r_fm"][mask_tr], color=colors_tr[e_cm], lw=2.3,
                         label=f"E_CM = {e_cm} MeV (Bounce at {tr['r_turning_point_fm']:.2f} fm)")
        axes[1].axhline(uv["r_bounce_wall_fm"], color="#f87171", lw=1.4, ls="--",
                        label=f"Multi-Quark Bounce Wall ({uv['r_bounce_wall_fm']:.2f} fm)")
        axes[1].axhline(uv["lambda_compton_quark_fm"], color="#c084fc", lw=1.3, ls="-.",
                        label=f"Quark Compton Scale λ_C,qB = {uv['lambda_compton_quark_fm']:.2f} fm")
        axes[1].axhline(uv["lambda_compton_nucleon_fm"], color="#a78bfa", lw=1.2, ls=":",
                        label=f"Point-Nucleon Scale λ_C,N = {uv['lambda_compton_nucleon_fm']:.2f} fm")
        axes[1].set_xlim(0, 35)
        axes[1].set_ylim(0.15, 3.65)
        axes[1].set_xlabel("Collision Time t [yoctoseconds, 10⁻²⁴ s]")
        axes[1].set_ylabel("Proton–Neutron Separation r(t) [fm]")
        axes[1].set_title("Simulated Head-On Proton–Neutron Collision Bounces")
        axes[1].grid(True)
        axes[1].legend(loc="upper right", fontsize=7.8)

        # Panel C: Planck-Scale Uncertainty Collapse Bounce vs Classical Singularity
        axes[2].plot(uv["t_collapse_norm"], uv["r_cufg_planck_bounce"], color="#38bdf8", lw=2.7,
                     label="CUFG Planck Phase-Space Bounce (Nonsingular)")
        axes[2].plot(uv["t_collapse_norm"], uv["r_classical_collapse"], color="#f87171", lw=2.0, ls="--",
                     label="Classical GR Collapse (r → 0 Singularity)")
        axes[2].axhline(uv["r_bounce_norm"], color="#facc15", lw=1.3, ls="-.",
                        label="Planck Saturation Radius (ρ = ρ_Planck)")
        axes[2].set_xlabel("Normalized Proper Time (t - t_bounce) / τ_0")
        axes[2].set_ylabel("Normalized Core Radius R(t) / R_0")
        axes[2].set_title("Planck-Scale Limit: Uncertainty Collapse Bounce")
        axes[2].set_ylim(0, 1.32)
        axes[2].grid(True)
        axes[2].legend(loc="upper right", fontsize=8.0)

        fig.tight_layout()
        p7 = out_dir / "fig7_uv_planck_and_nucleon_bounce.png"
        fig.savefig(p7)
        plt.close(fig)
        saved_paths.append(p7)

    # -------------------------------------------------------------------------
    # Figure 8: Subatomic & Atomic Particle Hierarchy from First-Principles Uncertainty
    # -------------------------------------------------------------------------
    if "subatomic" in results:
        sub = results["subatomic"]
        fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.9))

        # Panel A: Proton Variational Uncertainty Energy Minimization E(R)
        r_fm = sub["r_grid_fm"]
        axes[0].plot(r_fm, sub["e_tot_p"], color="#38bdf8", lw=2.7,
                     label="Total Proton Energy E_tot(R) (Min = 938.3 MeV)")
        axes[0].plot(r_fm, sub["e_kin_p"], color="#f87171", lw=1.9, ls="--",
                     label="Quark Heisenberg Uncertainty 3·x₀ħc/R (99% of Mass)")
        axes[0].plot(r_fm, sub["e_vac_p"], color="#34d399", lw=1.9, ls=":",
                     label="Vacuum Cavity Displacement (4/3)πR³B_QCD")
        idx_pmin = int(np.argmin(sub["e_tot_p"]))
        axes[0].scatter([r_fm[idx_pmin]], [sub["e_tot_p"][idx_pmin]], color="#facc15", s=75, zorder=6,
                        label=f"Equilibrium R_p = {r_fm[idx_pmin]:.2f} fm")
        axes[0].set_xlim(0.45, 1.75)
        axes[0].set_ylim(0, 2550)
        axes[0].set_xlabel("Confinement Cavity Radius R [fm]")
        axes[0].set_ylabel("Energy [MeV]")
        axes[0].set_title("Level 1: Proton Mass from Quark Uncertainty Balance")
        axes[0].grid(True)
        axes[0].legend(loc="upper right", fontsize=7.8)

        # Panel B: Complete Composite Hadron Spectrum (Baryon Octet/Decuplet & Mesons)
        if "elementary" in results:
            elem_h = results["elementary"]
            comb = elem_h["meson_spectrum"][:1] + elem_h["meson_spectrum"][2:] + elem_h["baryon_spectrum"]
            comb_sorted = sorted(comb, key=lambda x: x["obs_mev"])
            short_symbol_map = {
                "Pion": "π",
                "Kaon": "K",
                "Eta": "η",
                "Rho": "ρ/ω",
                "K*": "K*",
                "Nucleon": "N",
                "Eta-Prime": "η'",
                "Phi": "φ",
                "Lambda": "Λ",
                "Sigma": "Σ",
                "Delta": "Δ",
                "Xi": "Ξ",
                "Sigma*": "Σ*",
                "Xi*": "Ξ*",
                "Omega": "Ω⁻",
                "Charmed": "D",
                "Charm-Strange": "Dₛ",
                "D*": "D*",
                "D_s*": "Dₛ*",
                "Charmonium": "ηc",
                "J/Psi": "J/ψ",
                "Radial": "ψ(2S)",
                "Bottom": "B",
                "B*": "B*",
                "Bottom-Strange": "Bₛ",
                "B_s*": "Bₛ*",
                "Bottom-Charm": "Bc",
                "B_c*": "Bc*",
                "Bottomonium": "ηb",
                "Upsilon": "Υ(1S)",
            }
            h_labels = []
            for c in comb_sorted:
                nm = c["name"]
                if "Υ(2S)" in nm:
                    h_labels.append("Υ(2S)")
                elif "Υ(1S)" in nm:
                    h_labels.append("Υ(1S)")
                elif "η_b" in nm:
                    h_labels.append("ηb")
                elif "B_c*" in nm:
                    h_labels.append("Bc*")
                elif "B_c" in nm:
                    h_labels.append("Bc")
                elif "B_s*" in nm:
                    h_labels.append("Bₛ*")
                elif "B_s" in nm:
                    h_labels.append("Bₛ")
                elif "B*" in nm:
                    h_labels.append("B*")
                elif "Bottom B " in nm:
                    h_labels.append("B")
                elif "ψ(2S)" in nm:
                    h_labels.append("ψ(2S)")
                elif "J/ψ" in nm:
                    h_labels.append("J/ψ")
                elif "η_c" in nm:
                    h_labels.append("ηc")
                elif "D_s*" in nm:
                    h_labels.append("Dₛ*")
                elif "D_s" in nm:
                    h_labels.append("Dₛ")
                elif "D*(" in nm:
                    h_labels.append("D*")
                elif "Charmed D " in nm:
                    h_labels.append("D")
                elif "η'" in nm or "Eta-prime" in nm:
                    h_labels.append("η'")
                else:
                    first_tok = nm.split(" ")[0]
                    h_labels.append(short_symbol_map.get(first_tok, first_tok[:4]))
            x_idx = np.arange(len(h_labels))
            pred_h = [c["pred_mev"] for c in comb_sorted]
            obs_h = [c["obs_mev"] for c in comb_sorted]
            axes[1].scatter(x_idx, np.log10(obs_h), color="#facc15", s=50, marker="o", label="Observed PDG Mass", zorder=5)
            axes[1].scatter(x_idx, np.log10(pred_h), color="#38bdf8", s=50, marker="x", lw=2.0, label="Derived Layer-4 Mass", zorder=6)
            axes[1].set_xticks(x_idx)
            axes[1].set_xticklabels(h_labels, rotation=55, ha="right", fontsize=6.8)
            axes[1].set_ylabel("log₁₀(Hadron Mass [MeV/c²])")
            axes[1].set_title("31 Composite Hadrons: Mesons, Quarkonia & Baryons (Zero Bag Fit)")
            axes[1].grid(True)
            axes[1].legend(loc="upper left", fontsize=7.8)
        else:
            hads = sub["hadron_results"]
            names = [h["name"].split(" ")[0] for h in hads]
            x_idx = np.arange(len(names))
            pred_m = [h["predicted_mass_mev"] for h in hads]
            obs_m = [h["observed_mass_mev"] for h in hads]
            bare_m = [h["bare_quark_mass_mev"] for h in hads]
            width = 0.28
            axes[1].bar(x_idx - width, bare_m, width=width, color="#64748b", label="Bare Higgs Quark Mass (<2%)")
            axes[1].bar(x_idx, pred_m, width=width, color="#38bdf8", label="Derived Uncertainty Mass")
            axes[1].bar(x_idx + width, obs_m, width=width, color="#facc15", label="Observed PDG Mass")
            axes[1].set_xticks(x_idx)
            axes[1].set_xticklabels(names)
            axes[1].set_ylabel("Hadron Mass [MeV/c²]")
            axes[1].set_title(f"Hadron Spectrum & n-p Splitting (ΔM_np = {sub['delta_m_np_pred_mev']:.2f} MeV)")
            axes[1].grid(True, axis="y")
            axes[1].legend(loc="upper left", fontsize=7.8)

        # Panel C: Atomic Electron Shells (n=1, 2, 3) from Uncertainty Minimization
        r_ang = sub["r_angstrom"]
        colors_sh = {1: "#38bdf8", 2: "#34d399", 3: "#a78bfa"}
        for sh in sub["shells"]:
            n_s = sh["n"]
            axes[2].plot(r_ang, sub["shell_curves"][n_s], color=colors_sh[n_s], lw=2.3,
                         label=f"n={n_s} Shell (r={sh['r_pred_angstrom']:.2f} Å, E={sh['e_pred_ev']:.2f} eV)")
            axes[2].scatter([sh["r_pred_angstrom"]], [sh["e_pred_ev"]], color="#facc15", s=55, zorder=6)
        axes[2].axhline(0.0, color="#94a3b8", lw=1.0, ls=":")
        axes[2].set_xlim(0.2, 6.0)
        axes[2].set_ylim(-15.5, 5.0)
        axes[2].set_xlabel("Electron Orbital Radius r [Å]")
        axes[2].set_ylabel("Effective Shell Energy E_n(r) [eV]")
        axes[2].set_title("Level 4: Atomic Shells from Heisenberg Uncertainty")
        axes[2].grid(True)
        axes[2].legend(loc="lower right", fontsize=7.8)

        fig.tight_layout()
        p8 = out_dir / "fig8_subatomic_particle_hierarchy.png"
        fig.savefig(p8)
        plt.close(fig)
        saved_paths.append(p8)

    # =========================================================================
    # FIGURE 9: Elementary Point-Particle Mass Spectrum, Z_3 Geometry & 3x3 Mixing
    # =========================================================================
    if "elementary" in results:
        elem = results["elementary"]
        fig, axes_grid = plt.subplots(2, 2, figsize=(16.5, 10.5))
        axes = axes_grid.flatten()

        # Panel A: Full 15-Particle Massive Elementary Spectrum (Log10 Scale, eV to GeV)
        nu = elem["neutrinos"]
        part_labels = [
            "ν₁", "ν₂", "ν₃",
            "e⁻", "μ⁻", "τ⁻",
            "u", "d", "s", "c", "b",
            "W±", "Z⁰", "H⁰", "t"
        ]
        pred_ev = [
            nu["m_nu1_ev"], nu["m_nu2_ev"], nu["m_nu3_ev"],
            elem["leptons"][0]["pred_mev"] * 1e6,
            elem["leptons"][1]["pred_mev"] * 1e6,
            elem["leptons"][2]["pred_mev"] * 1e6,
            elem["up_quarks"][0]["pred_mev"] * 1e6,
            elem["down_quarks"][0]["pred_mev"] * 1e6,
            elem["down_quarks"][1]["pred_mev"] * 1e6,
            elem["up_quarks"][1]["pred_mev"] * 1e6,
            elem["down_quarks"][2]["pred_mev"] * 1e6,
            elem["bosons"][2]["pred_mev"] * 1e6,
            elem["bosons"][3]["pred_mev"] * 1e6,
            elem["bosons"][4]["pred_mev"] * 1e6,
            elem["up_quarks"][2]["pred_mev"] * 1e6,
        ]
        obs_ev = [
            nu["m_nu1_ev"],
            np.sqrt(nu["m_nu1_ev"]**2 + nu["dm21_sq_obs_ev2"]),
            np.sqrt(nu["m_nu1_ev"]**2 + nu["dm31_sq_obs_ev2"]),
            elem["leptons"][0]["obs_mev"] * 1e6,
            elem["leptons"][1]["obs_mev"] * 1e6,
            elem["leptons"][2]["obs_mev"] * 1e6,
            elem["up_quarks"][0]["obs_mev"] * 1e6,
            elem["down_quarks"][0]["obs_mev"] * 1e6,
            elem["down_quarks"][1]["obs_mev"] * 1e6,
            elem["up_quarks"][1]["obs_mev"] * 1e6,
            elem["down_quarks"][2]["obs_mev"] * 1e6,
            elem["bosons"][2]["obs_mev"] * 1e6,
            elem["bosons"][3]["obs_mev"] * 1e6,
            elem["bosons"][4]["obs_mev"] * 1e6,
            elem["up_quarks"][2]["obs_mev"] * 1e6,
        ]
        x_p = np.arange(len(part_labels))
        axes[0].scatter(x_p, np.log10(obs_ev), color="#facc15", s=75, marker="o", label="Observed (PDG / NuFIT)", zorder=5)
        axes[0].scatter(x_p, np.log10(pred_ev), color="#38bdf8", s=70, marker="x", lw=2.3, label="Derived 3+1D Phase-Space", zorder=6)
        axes[0].set_xticks(x_p)
        axes[0].set_xticklabels(part_labels)
        axes[0].set_ylabel("log₁₀(Mass [eV/c²])")
        eg = elem["emergent_uncertainty_scale"]
        axes[0].set_title(f"15 Massive Elementary Particles & α_unc^(2-body) = {eg['alpha_unc_2body']:.3e} (ℓ_UV Err {eg['l_uv_rel_err_ppm']:.2f} ppm)")
        axes[0].grid(True)
        axes[0].legend(loc="upper left", fontsize=8.5)

        # Panel B: Z_3 Phase-Space Function z(theta)^2 = (1 + sqrt(2) cos(theta))^2
        theta_grid = np.linspace(0, 2.0 * np.pi, 600)
        z_sq_curve = (1.0 + np.sqrt(2.0) * np.cos(theta_grid)) ** 2
        axes[1].plot(theta_grid, z_sq_curve, color="#94a3b8", lw=2.0, label="z(θ)² = [1 + √2 cos(θ)]² (R=1)")
        axes[1].axvline(3.0 * np.pi / 4.0, color="#f43f5e", ls="--", lw=1.2, label="Massless Node θ_crit = 3π/4")
        axes[1].axvline(5.0 * np.pi / 4.0, color="#f43f5e", ls="--", lw=1.2)

        delta_l = 2.0 / 9.0
        th_l = [(2.0 * np.pi / 3.0 + delta_l + 2.0 * np.pi * k / 3.0) % (2.0 * np.pi) for k in range(3)]
        z2_l = [(1.0 + np.sqrt(2.0) * np.cos(t)) ** 2 for t in th_l]
        axes[1].scatter(th_l, z2_l, color="#38bdf8", s=70, zorder=6, label="Charged Leptons (δ_l = 2/9)")

        th_d = [(3.0 * np.pi / 4.0 - 2.0 / 27.0 + 2.0 * np.pi * k / 3.0) % (2.0 * np.pi) for k in range(3)]
        z2_d = [(1.0 + np.sqrt(2.0) * np.cos(t)) ** 2 for t in th_d]
        axes[1].scatter(th_d, z2_d, color="#34d399", s=70, marker="^", zorder=6, label="Down Quarks (Δδ_d = 2/27)")

        axes[1].set_xlim(0, 2.0 * np.pi)
        axes[1].set_ylim(-0.35, 7.85)
        axes[1].set_xlabel("Generation Phase Angle θ_k [rad]")
        axes[1].set_ylabel("Dimensionless Eigenvalue Weight z_k²")
        axes[1].set_title("Z₃ Equipartition (Q = 2/3) & Massless Boundary")
        axes[1].grid(True)
        axes[1].legend(loc="upper right", fontsize=8.0)

        # Panel C: Quark Sector Unscreened (Constituent/Pole) vs Color-Screened (MS-bar) vs PDG
        q_names = ["u", "d", "s", "c", "b", "t"]
        q_unscreened = [
            elem["up_quarks"][0]["pred_unscreened_mev"],
            elem["down_quarks"][0]["pred_unscreened_mev"],
            elem["down_quarks"][1]["pred_unscreened_mev"],
            elem["up_quarks"][1]["pred_unscreened_mev"],
            elem["down_quarks"][2]["pred_unscreened_mev"],
            elem["up_quarks"][2]["pred_unscreened_mev"],
        ]
        q_screened = [
            elem["up_quarks"][0]["pred_mev"],
            elem["down_quarks"][0]["pred_mev"],
            elem["down_quarks"][1]["pred_mev"],
            elem["up_quarks"][1]["pred_mev"],
            elem["down_quarks"][2]["pred_mev"],
            elem["up_quarks"][2]["pred_mev"],
        ]
        q_obs = [
            elem["up_quarks"][0]["obs_mev"],
            elem["down_quarks"][0]["obs_mev"],
            elem["down_quarks"][1]["obs_mev"],
            elem["up_quarks"][1]["obs_mev"],
            elem["down_quarks"][2]["obs_mev"],
            elem["up_quarks"][2]["obs_mev"],
        ]
        xq = np.arange(len(q_names))
        w = 0.26
        axes[2].bar(xq - w, np.log10(q_unscreened), width=w, color="#64748b", label="Unscreened R=1 (Constituent/Pole)")
        axes[2].bar(xq, np.log10(q_screened), width=w, color="#38bdf8", label="Color-Casimir Screened (C_F=4/3)")
        axes[2].bar(xq + w, np.log10(q_obs), width=w, color="#facc15", label="Observed PDG Mass")
        axes[2].set_xticks(xq)
        axes[2].set_xticklabels(q_names)
        axes[2].set_ylabel("log₁₀(Quark Mass [MeV/c²])")
        axes[2].set_title("Quark Sector: Unscreened vs SU(3)_C Screened (F11)")
        axes[2].grid(True, axis="y")
        axes[2].legend(loc="upper left", fontsize=8.0)

        # Panel D: Full 3x3 CKM and 3x3 PMNS Weak Mixing Matrix Elements (Derived vs Observed)
        mix = elem["mixing"]
        ckm_labels = ["|V_ud|", "|V_us|", "|V_ub|", "|V_cd|", "|V_cs|", "|V_cb|", "|V_td|", "|V_ts|", "|V_tb|"]
        ckm_pred_flat = np.array(mix["v_ckm_pred"]).flatten()
        ckm_obs_flat = np.array(mix["v_ckm_obs"]).flatten()

        pmns_labels = ["|U_e1|", "|U_e2|", "|U_e3|", "|U_μ1|", "|U_μ2|", "|U_μ3|", "|U_τ1|", "|U_τ2|", "|U_τ3|"]
        pmns_pred_flat = np.array(mix["u_pmns_pred"]).flatten()
        pmns_obs_flat = np.array(mix["u_pmns_obs"]).flatten()

        x_m = np.arange(9)
        axes[3].plot(x_m, np.log10(ckm_obs_flat), "o-", color="#facc15", lw=1.6, label="Observed 3x3 CKM (PDG)")
        axes[3].plot(x_m, np.log10(ckm_pred_flat), "x--", color="#38bdf8", lw=2.0, markersize=8, label="Derived 3x3 CKM (F13-F16)")
        axes[3].plot(x_m, np.log10(pmns_obs_flat), "s-", color="#a78bfa", lw=1.6, label="Observed 3x3 PMNS (NuFIT)")
        axes[3].plot(x_m, np.log10(pmns_pred_flat), "^--", color="#34d399", lw=2.0, markersize=7, label="Derived 3x3 PMNS (F17-F19)")
        axes[3].set_xticks(x_m)
        axes[3].set_xticklabels([f"{c}\n{p}" for c, p in zip(ckm_labels, pmns_labels)], fontsize=7.8)
        axes[3].set_ylim(-3.05, 0.25)
        axes[3].set_ylabel("log₁₀(|Matrix Element|)")
        axes[3].set_title("Derived vs. Observed 3×3 CKM & PMNS Mixing Matrices")
        axes[3].grid(True)
        axes[3].legend(loc="lower left", fontsize=7.8, ncol=2)

        fig.tight_layout()
        p9 = out_dir / "fig9_elementary_particle_masses.png"
        fig.savefig(p9)
        plt.close(fig)
        saved_paths.append(p9)

    # -------------------------------------------------------------------------
    # Figure 10: CAMB/CLASS-Architecture Multipole Uncertainty-Boltzmann Hierarchy
    #            Across 2 <= l <= 2500 (TT, EE, TE Power Spectra vs. Planck 2018)
    # -------------------------------------------------------------------------
    if "camb_class_l2500" in results["cosmology"]:
        cb = results["cosmology"]["camb_class_l2500"]
        l_grid = cb["l_multipole_grid"]
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

        # Panel A: Temperature Power Spectrum D_l^TT across 2 <= l <= 2500 (7 Peaks)
        axes[0].plot(
            l_grid,
            cb["dl_tt_full"],
            color="#38bdf8",
            lw=2.2,
            label="CUFG Uncertainty-Boltzmann (2π Cyclic Horizon, Zero G & Zero CDM)",
        )
        axes[0].plot(
            l_grid,
            cb["dl_tt_no_2pi_full"],
            color="#f87171",
            lw=1.6,
            ls="--",
            label="Naive Baryon-Only (Without 2π Cyclic Horizon Overlap)",
        )
        tt_l_obs = [10.0, cb["l_trough1_obs"]] + [p["l_obs"] for p in cb["tt_peaks_l2500"]]
        tt_dl_obs = [cb["dl_sw_l10_obs_uk2"], cb["dl_trough1_obs_uk2"]] + [
            p["dl_tt_obs_uk2"] for p in cb["tt_peaks_l2500"]
        ]
        axes[0].scatter(
            tt_l_obs,
            tt_dl_obs,
            color="#facc15",
            s=55,
            zorder=6,
            label="Planck 2018 (SW Plateau, Trough 1 & Peaks m=1..7)",
        )
        axes[0].set_xlim(2, 2500)
        axes[0].set_ylim(0, 6850)
        axes[0].set_xlabel("Multipole Moment ℓ")
        axes[0].set_ylabel(r"$\mathcal{D}_\ell^{TT} = \ell(\ell+1)C_\ell^{TT}/(2\pi)$ [$\mu\mathrm{K}^2$]")
        axes[0].set_title("CMB Temperature Spectrum (2 ≤ ℓ ≤ 2500, Peaks 1–7)")
        axes[0].grid(True)
        axes[0].legend(loc="upper right", fontsize=7.6)

        # Panel B: E-Mode Polarization Power Spectrum D_l^EE across 2 <= l <= 2500 (6 Peaks)
        axes[1].plot(
            l_grid,
            cb["dl_ee_full"],
            color="#34d399",
            lw=2.2,
            label=r"CUFG Spin-2 E-Mode Polarization ($c_E = \sqrt{6}/10,\ \phi_E = 2/9$)",
        )
        ee_l_obs = [p["l_obs"] for p in cb["ee_peaks_l2500"]]
        ee_dl_obs = [p["dl_ee_obs_uk2"] for p in cb["ee_peaks_l2500"]]
        axes[1].scatter(
            ee_l_obs,
            ee_dl_obs,
            color="#facc15",
            s=55,
            zorder=6,
            label="Planck 2018 E-Mode Polarization Peaks (m=1..6)",
        )
        axes[1].set_xlim(2, 2500)
        axes[1].set_ylim(0, 50)
        axes[1].set_xlabel("Multipole Moment ℓ")
        axes[1].set_ylabel(r"$\mathcal{D}_\ell^{EE} = \ell(\ell+1)C_\ell^{EE}/(2\pi)$ [$\mu\mathrm{K}^2$]")
        axes[1].set_title("CMB E-Mode Polarization Spectrum (2 ≤ ℓ ≤ 2500, Peaks 1–6)")
        axes[1].grid(True)
        axes[1].legend(loc="upper right", fontsize=7.8)

        # Panel C: Temperature-Polarization Cross-Spectrum D_l^TE across 2 <= l <= 2500
        axes[2].plot(
            l_grid,
            cb["dl_te_full"],
            color="#a78bfa",
            lw=2.0,
            label="CUFG TE Cross-Spectrum (3D/4D Monopole-Quadrupole Overlap)",
        )
        axes[2].axhline(0.0, color="#94a3b8", lw=1.0, ls=":")
        te_l_obs = [p["l_obs"] for p in cb["te_extrema_l2500"]]
        te_dl_obs = [p["dl_te_obs_uk2"] for p in cb["te_extrema_l2500"]]
        axes[2].scatter(
            te_l_obs,
            te_dl_obs,
            color="#facc15",
            s=55,
            zorder=6,
            label="Planck 2018 TE Extrema (Anticorrelation & Peaks n=1..6)",
        )
        axes[2].set_xlim(2, 2500)
        axes[2].set_ylim(-165, 185)
        axes[2].set_xlabel("Multipole Moment ℓ")
        axes[2].set_ylabel(r"$\mathcal{D}_\ell^{TE} = \ell(\ell+1)C_\ell^{TE}/(2\pi)$ [$\mu\mathrm{K}^2$]")
        axes[2].set_title("CMB TE Cross-Power Spectrum (2 ≤ ℓ ≤ 2500)")
        axes[2].grid(True)
        axes[2].legend(loc="upper right", fontsize=7.8)

        fig.tight_layout()
        p10 = out_dir / "fig10_cmb_boltzmann_hierarchy_l2500.png"
        fig.savefig(p10)
        plt.close(fig)
        saved_paths.append(p10)

    return saved_paths
