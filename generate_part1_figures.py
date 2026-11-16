#!/usr/bin/env python3
"""
generate_part1_figures.py - Generates verified publication figures & Plotly interactive
components for Part 1 (Projects 01-06).
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Add workspace to path for shared_utils
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from shared_utils.physics_helpers import (
    simulate_wind_wake_field,
    solve_power_grid_swing,
    simulate_battery_impedance_spectrum,
    solve_2d_heat_conduction,
    simulate_solar_irradiance_clearsky,
    simulate_pipeline_pressure_wave,
    heat_equation_residual,
    mass_conservation_error
)
from shared_utils.plot_helpers import (
    set_academic_matplotlib_style,
    save_matplotlib_figure,
    export_plotly_html_snippet
)

def run():
    set_academic_matplotlib_style()
    figures_dir = root_dir / "dist" / "assets" / "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    print("Generating Part 1 Figures & Interactive Diagnostics...")

    # =========================================================================
    # Project 01: Wind Turbine Wake Modeling
    # =========================================================================
    print("  -> Project 01: Wind Wake Deficit...")
    x = np.linspace(-100, 900, 150)
    y = np.linspace(-200, 200, 100)
    X, Y = np.meshgrid(x, y)
    turbines = [(0.0, 0.0), (350.0, -40.0), (700.0, 20.0)]
    u_field, v_field = simulate_wind_wake_field(X, Y, turbines, u_inf=12.0)
    
    # 1. Matplotlib Figure
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    cp = ax.contourf(X, Y, u_field, levels=30, cmap="viridis")
    cbar = fig.colorbar(cp, ax=ax)
    cbar.set_label("Velocity $u(x, y)$ (m/s)")
    for idx, (tx, ty) in enumerate(turbines):
        ax.plot([tx, tx], [ty - 40, ty + 40], color="red", linewidth=3, label="Turbine Rotor" if idx==0 else "")
        ax.scatter(tx, ty, color="black", s=40, zorder=5)
    ax.set_title("Jensen-Gaussian Wake Deficit & Turbulent Momentum Recovery", fontweight='bold')
    ax.set_xlabel("Downwind Distance $x$ (m)")
    ax.set_ylabel("Crosswind Coordinate $y$ (m)")
    ax.legend(loc="upper right")
    save_matplotlib_figure(fig, str(figures_dir / "01_wind_wake_contour.png"))

    # 2. Interactive Plotly
    plotly_fig1 = go.Figure(data=go.Contour(
        z=u_field,
        x=x,
        y=y,
        colorscale='Viridis',
        colorbar=dict(title='Velocity (m/s)')
    ))
    for tx, ty in turbines:
        plotly_fig1.add_shape(
            type="line", x0=tx, y0=ty-40, x1=tx, y1=ty+40,
            line=dict(color="Red", width=4)
        )
    plotly_fig1.update_layout(
        title="Interactive 2D Wind Velocity Deficit Field [Inspect Coordinates]",
        xaxis_title="Downwind Distance x (m)",
        yaxis_title="Crosswind Coordinate y (m)",
        height=450
    )
    with open(figures_dir / "01_wind_wake_interactive.html", "w") as f:
        f.write(export_plotly_html_snippet(plotly_fig1, div_id="plot-wind-wake"))

    # =========================================================================
    # Project 02: Smart-Grid Load Dynamics
    # =========================================================================
    print("  -> Project 02: Smart Grid Dynamics...")
    t_eval = np.linspace(0, 10, 500)
    grid_sol = solve_power_grid_swing((0, 10), t_eval, fault_start=2.0, fault_duration=0.18)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 5.0), sharex=True)
    ax1.plot(grid_sol['t'], grid_sol['delta'], color="#1e40af", linewidth=2, label="Rotor Angle $\\delta(t)$")
    ax1.axvspan(2.0, 2.18, color="#fee2e2", alpha=0.8, label="Three-Phase Bus Fault (180ms)")
    ax1.set_ylabel("Angle $\\delta$ (rad)")
    ax1.set_title("Synchronous Generator Swing Response to Fault Disruption", fontweight='bold')
    ax1.legend(loc="upper right")

    ax2.plot(grid_sol['t'], grid_sol['frequency_hz'], color="#059669", linewidth=2, label="Bus Frequency $f(t)$")
    ax2.axhline(50.0, color="gray", linestyle="--", alpha=0.7)
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right")
    save_matplotlib_figure(fig, str(figures_dir / "02_grid_frequency_dynamics.png"))

    plotly_fig2 = go.Figure()
    plotly_fig2.add_trace(go.Scatter(x=grid_sol['t'], y=grid_sol['frequency_hz'], mode='lines', name='Grid Frequency (Hz)', line=dict(color='#059669', width=2.5)))
    plotly_fig2.add_vrect(x0=2.0, x1=2.18, fillcolor="red", opacity=0.15, line_width=0, annotation_text="Grid Fault Event")
    plotly_fig2.update_layout(title="Interactive Grid Frequency Recovery Curve", xaxis_title="Time (seconds)", yaxis_title="Frequency (Hz)", height=400)
    with open(figures_dir / "02_grid_interactive.html", "w") as f:
        f.write(export_plotly_html_snippet(plotly_fig2, div_id="plot-grid-freq"))

    # =========================================================================
    # Project 03: Battery Degradation & EIS
    # =========================================================================
    print("  -> Project 03: Battery Impedance Spectroscopy...")
    freqs = np.logspace(4, -2, 70)
    # Fresh cell vs Degraded cell
    z_real_fresh, z_imag_fresh = simulate_battery_impedance_spectrum(freqs, r_charge_transfer=0.05, sigma_warburg=0.03)
    z_real_aged, z_imag_aged = simulate_battery_impedance_spectrum(freqs, r_charge_transfer=0.18, sigma_warburg=0.08)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(z_real_fresh, z_imag_fresh, 'o-', color="#0284c7", label="Fresh Cell (100% SOH, R_ct=0.05 $\\Omega$)")
    ax.plot(z_real_aged, z_imag_aged, 's-', color="#e11d48", label="Aged Cell (81% SOH, R_ct=0.18 $\\Omega$)")
    ax.set_title("Electrochemical Impedance Spectroscopy (EIS) Nyquist Projection", fontweight='bold')
    ax.set_xlabel("Real Impedance $Z'(\\omega)$ ($\\Omega$)")
    ax.set_ylabel("Negative Imaginary Impedance $-Z''(\\omega)$ ($\\Omega$)")
    ax.legend(loc="upper left")
    save_matplotlib_figure(fig, str(figures_dir / "03_battery_nyquist_degradation.png"))

    plotly_fig3 = go.Figure()
    plotly_fig3.add_trace(go.Scatter(x=z_real_fresh, y=z_imag_fresh, mode='lines+markers', name='Fresh Cell (100% SOH)', marker=dict(color='#0284c7')))
    plotly_fig3.add_trace(go.Scatter(x=z_real_aged, y=z_imag_aged, mode='lines+markers', name='Aged Cell (81% SOH)', marker=dict(color='#e11d48')))
    plotly_fig3.update_layout(title="Interactive Nyquist Plot: EIS SOH Degradation Arc", xaxis_title="Z' Real (Ohms)", yaxis_title="-Z'' Imaginary (Ohms)", height=420)
    with open(figures_dir / "03_battery_interactive.html", "w") as f:
        f.write(export_plotly_html_snippet(plotly_fig3, div_id="plot-battery-nyquist"))

    # =========================================================================
    # Project 04: Nuclear Reactor Core Temperature
    # =========================================================================
    print("  -> Project 04: Nuclear Core Heat Conduction...")
    X_core, Y_core, T_core = solve_2d_heat_conduction(nx=45, ny=45, alpha=0.02, core_power_source=2800.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cs = ax.contourf(X_core, Y_core, T_core, levels=30, cmap="inferno")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Fuel Rod Temp $T(x,y)$ (K)")
    ax.set_title("Nuclear Core Steady-State Conduction & Hotspot Distribution", fontweight='bold')
    ax.set_xlabel("Normalized Radial Locus $x/L$")
    ax.set_ylabel("Normalized Radial Locus $y/L$")
    save_matplotlib_figure(fig, str(figures_dir / "04_nuclear_core_temp_field.png"))

    plotly_fig4 = go.Figure(data=go.Contour(
        z=T_core,
        x=X_core[0, :],
        y=Y_core[:, 0],
        colorscale='Inferno',
        colorbar=dict(title='Temp (K)')
    ))
    plotly_fig4.update_layout(title="Interactive 2D Core Thermal Profile [Hover to inspect hotspot]", height=430)
    with open(figures_dir / "04_nuclear_interactive.html", "w") as f:
        f.write(export_plotly_html_snippet(plotly_fig4, div_id="plot-nuclear-core"))

    # =========================================================================
    # Project 05: Solar Photovoltaic Irradiance
    # =========================================================================
    print("  -> Project 05: Solar Irradiance Attenuation...")
    hours = np.linspace(6.0, 18.0, 100)
    zeniths = 90.0 - 65.0 * np.sin((hours - 6.0) / 12.0 * np.pi)
    ghi_clean = simulate_solar_irradiance_clearsky(zeniths, aerosol_optical_depth=0.06)
    ghi_haze = simulate_solar_irradiance_clearsky(zeniths, aerosol_optical_depth=0.35)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(hours, ghi_clean, color="#f59e0b", linewidth=2.5, label="Clean Atmospheric Atmosphere (AOD=0.06)")
    ax.plot(hours, ghi_haze, '--', color="#64748b", linewidth=2.0, label="Urban Haze / Aerosol Plume (AOD=0.35)")
    ax.set_title("Diurnal Global Horizontal Irradiance (GHI) Beer-Lambert Attenuation", fontweight='bold')
    ax.set_xlabel("Local Solar Time (Hours)")
    ax.set_ylabel("Global Horizontal Irradiance $GHI$ (W/m²)")
    ax.legend(loc="upper right")
    save_matplotlib_figure(fig, str(figures_dir / "05_solar_irradiance_clearsky.png"))

    plotly_fig5 = go.Figure()
    plotly_fig5.add_trace(go.Scatter(x=hours, y=ghi_clean, mode='lines', name='Clear Atmosphere (AOD=0.06)', line=dict(color='#f59e0b', width=3)))
    plotly_fig5.add_trace(go.Scatter(x=hours, y=ghi_haze, mode='lines', name='Hazy / Aerosol (AOD=0.35)', line=dict(color='#64748b', dash='dash', width=2)))
    plotly_fig5.update_layout(title="Interactive Diurnal Solar Irradiance Profile", xaxis_title="Hour of Day", yaxis_title="GHI (W/m²)", height=400)
    with open(figures_dir / "05_solar_interactive.html", "w") as f:
        f.write(export_plotly_html_snippet(plotly_fig5, div_id="plot-solar-irradiance"))

    # =========================================================================
    # Project 06: Pipeline Leak Transient Wave
    # =========================================================================
    print("  -> Project 06: Pipeline Acoustic Leak Wave...")
    pipe_x, pipe_t, pipe_p = simulate_pipeline_pressure_wave(length_m=2000.0, leak_location_m=750.0)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    cp_pipe = ax.imshow(pipe_p, aspect='auto', extent=[0, 2000, pipe_t[-1], 0], cmap='plasma')
    cbar = fig.colorbar(cp_pipe, ax=ax)
    cbar.set_label("Transient Pressure $P(x, t)$ (bar)")
    ax.axvline(750, color='cyan', linestyle='--', linewidth=2, label='Pipeline Leak Orifice (x = 750m)')
    ax.set_title("Water-Hammer Acoustic Pressure Wave Reflection along 2km Pipeline", fontweight='bold')
    ax.set_xlabel("Distance along Pipeline $x$ (m)")
    ax.set_ylabel("Time $t$ (seconds)")
    ax.legend(loc="upper right")
    save_matplotlib_figure(fig, str(figures_dir / "06_pipeline_pressure_wave.png"))

    plotly_fig6 = go.Figure(data=go.Heatmap(
        z=pipe_p,
        x=pipe_x,
        y=pipe_t,
        colorscale='Plasma',
        colorbar=dict(title='Pressure (bar)')
    ))
    plotly_fig6.add_vline(x=750, line_width=3, line_dash="dash", line_color="cyan", annotation_text="Leak (750m)")
    plotly_fig6.update_layout(title="Interactive Water-Hammer Acoustic Wave Space-Time Diagram", xaxis_title="Pipeline Distance (m)", yaxis_title="Time (s)", height=430)
    with open(figures_dir / "06_pipeline_interactive.html", "w") as f:
        f.write(export_plotly_html_snippet(plotly_fig6, div_id="plot-pipeline-wave"))

    print("Successfully generated all Part 1 figures and interactive embeds!")

if __name__ == "__main__":
    run()
