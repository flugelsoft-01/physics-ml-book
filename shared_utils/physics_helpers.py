"""
Physics Helpers: Conservation Law Validators, PDE Residuals, and Synthetic Data Generators.
"""
from typing import Tuple, Dict, Any, Callable
import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import laplace


# ==========================================
# 1. Differential Operators & PDE Residuals
# ==========================================

def compute_laplacian_2d(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Computes second-order central difference 2D Laplacian."""
    d2u_dx2 = (np.roll(field, -1, axis=1) - 2 * field + np.roll(field, 1, axis=1)) / (dx ** 2)
    d2u_dy2 = (np.roll(field, -1, axis=0) - 2 * field + np.roll(field, 1, axis=0)) / (dy ** 2)
    # Mask boundaries to avoid periodic wrap artifacts
    res = d2u_dx2 + d2u_dy2
    res[0, :] = 0.0
    res[-1, :] = 0.0
    res[:, 0] = 0.0
    res[:, -1] = 0.0
    return res


def compute_divergence_2d(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Computes divergence div(U) = du/dx + dv/dy using central differences."""
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    return du_dx + dv_dy


def heat_equation_residual(u: np.ndarray, alpha: float, dx: float, dy: float, source: np.ndarray = None) -> float:
    """
    Computes steady-state heat equation residual: || alpha * nabla^2(u) + q ||_2.
    For steady conduction, residual should be close to 0 in physical equilibrium.
    """
    lap = compute_laplacian_2d(u, dx, dy)
    lhs = alpha * lap
    if source is not None:
        lhs += source
    # Interior nodes residual norm
    interior = lhs[1:-1, 1:-1]
    return float(np.sqrt(np.mean(interior ** 2)))


def mass_conservation_error(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> float:
    """Incompressible continuity residual || div(U) ||_2."""
    div = compute_divergence_2d(u, v, dx, dy)
    interior = div[1:-1, 1:-1]
    return float(np.sqrt(np.mean(interior ** 2)))


# ==========================================
# 2. Synthetic Physics Solvers & Simulators
# ==========================================

def simulate_wind_wake_field(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    turbine_positions: list,
    u_inf: float = 12.0,
    rotor_diameter: float = 80.0,
    thrust_coeff: float = 0.8,
    entrainment_k: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates a 2D wind turbine wake velocity deficit field using the
    analytical Jensen-Gaussian wake formulation.
    """
    u_field = np.full(grid_x.shape, u_inf, dtype=np.float64)
    v_field = np.zeros_like(u_field)

    for (tx, ty) in turbine_positions:
        dx = grid_x - tx
        dy = grid_y - ty
        
        # Downwind mask
        downwind = dx > 0
        r_wake = (rotor_diameter / 2.0) + entrainment_k * np.maximum(0.0, dx)
        
        # Gaussian deficit
        axial_induction = 0.5 * (1.0 - np.sqrt(max(0.01, 1.0 - thrust_coeff)))
        sigma = np.maximum(r_wake / 2.5, 1.0)
        
        raw_deficit = (2.0 * axial_induction * (rotor_diameter / (2.0 * sigma)) ** 2 * 
                       np.exp(-0.5 * (dy / sigma) ** 2))
        deficit = np.where(downwind, raw_deficit, 0.0)
        deficit = np.nan_to_num(deficit, nan=0.0)
        deficit = np.clip(deficit, 0.0, 0.85 * u_inf)
        u_field -= deficit

    # Add small physical turbulent fluctuation
    np.random.seed(42)
    noise = np.random.normal(0, 0.15, size=grid_x.shape)
    u_field = np.clip(u_field + noise, 0.5, u_inf * 1.2)
    
    return u_field, v_field


def solve_power_grid_swing(
    t_span: Tuple[float, float],
    t_eval: np.ndarray,
    p_mech: float = 1.0,
    p_max: float = 1.2,
    damping_d: float = 0.2,
    inertia_m: float = 0.1,
    fault_start: float = 1.0,
    fault_duration: float = 0.2,
    initial_state: Tuple[float, float] = (0.2, 0.0)
) -> Dict[str, np.ndarray]:
    """
    Simulates synchronous generator swing ODE:
    M * d2(delta)/dt2 + D * d(delta)/dt = P_mech - P_elec
    where P_elec = P_max * sin(delta) (or 0 during electrical fault).
    """
    def swing_ode(t, y):
        delta, omega = y
        # Electrical power output (drops during severe three-phase bus fault)
        if fault_start <= t <= (fault_start + fault_duration):
            p_elec = 0.15 * p_max * np.sin(delta)
        else:
            p_elec = p_max * np.sin(delta)
        
        d_delta = omega
        d_omega = (p_mech - p_elec - damping_d * omega) / inertia_m
        return [d_delta, d_omega]

    sol = solve_ivp(swing_ode, t_span, initial_state, t_eval=t_eval, method='RK45')
    return {
        't': sol.t,
        'delta': sol.y[0],
        'omega': sol.y[1],
        'frequency_hz': 50.0 + sol.y[1] / (2.0 * np.pi)
    }


def simulate_battery_impedance_spectrum(
    frequencies: np.ndarray,
    r_electrolyte: float = 0.025,
    r_charge_transfer: float = 0.08,
    c_double_layer: float = 1.5e-3,
    sigma_warburg: float = 0.04
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic Electrochemical Impedance Spectroscopy (EIS) Nyquist data
    based on a Randles equivalent circuit:
    Z(w) = R_el + [ (1 / R_ct + j*w*C_dl)^-1 ] + (1 - j) * sigma_w / sqrt(w)
    """
    omega = 2.0 * np.pi * frequencies
    # Double layer & charge transfer parallel branch
    y_parallel = (1.0 / r_charge_transfer) + 1j * omega * c_double_layer
    z_parallel = 1.0 / y_parallel
    
    # Warburg impedance (low frequency diffusion tail)
    z_warburg = (sigma_warburg / np.sqrt(omega)) * (1.0 - 1j)
    
    z_total = r_electrolyte + z_parallel + z_warburg
    z_real = np.real(z_total)
    z_imag = -np.imag(z_total)  # Standard Nyquist displays -Z_imag on vertical axis
    return z_real, z_imag


def solve_2d_heat_conduction(
    nx: int = 50,
    ny: int = 50,
    alpha: float = 0.015,
    t_boundary_top: float = 300.0,
    t_boundary_bottom: float = 550.0,
    t_boundary_left: float = 300.0,
    t_boundary_right: float = 300.0,
    core_power_source: float = 2500.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves steady-state 2D Poisson thermal distribution for a nuclear/thermal core:
    -alpha * (d2T/dx2 + d2T/dy2) = Q(x, y)
    """
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Internal localized cylindrical core heat source
    r2 = (X - 0.5)**2 + (Y - 0.5)**2
    Q = core_power_source * np.exp(-r2 / (2 * (0.15**2)))
    
    T = np.full((ny, nx), 300.0)
    T[-1, :] = t_boundary_top
    T[0, :] = t_boundary_bottom
    T[:, 0] = t_boundary_left
    T[:, -1] = t_boundary_right

    # Successive Over-Relaxation (SOR) solver
    omega = 1.82
    max_iter = 3000
    tol = 1e-5
    
    for _ in range(max_iter):
        T_old = T.copy()
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                t_star = 0.25 * (T[i+1, j] + T[i-1, j] + T[i, j+1] + T[i, j-1] + (dx * dy / alpha) * Q[i, j])
                T[i, j] = (1.0 - omega) * T[i, j] + omega * t_star
        if np.max(np.abs(T - T_old)) < tol:
            break
            
    return X, Y, T


def simulate_solar_irradiance_clearsky(
    zenith_angles_deg: np.ndarray,
    altitude_m: float = 150.0,
    i_solar_constant: float = 1367.0,
    ozone_cm: float = 0.35,
    water_vapor_cm: float = 1.5,
    aerosol_optical_depth: float = 0.08
) -> np.ndarray:
    """
    Computes Direct Normal Irradiance (DNI) via Beer-Lambert atmospheric attenuation:
    DNI = I_0 * exp( - tau * AM )
    where AM is relative air mass and tau combines Rayleigh, ozone, water vapor, and aerosols.
    """
    zenith_rad = np.radians(zenith_angles_deg)
    cos_z = np.maximum(np.cos(zenith_rad), 0.001)
    # Kasten-Young air mass formulation
    air_mass = 1.0 / (cos_z + 0.50572 * ((96.07995 - zenith_angles_deg) ** (-1.6364)))
    air_mass = np.where(zenith_angles_deg > 89.0, 0.0, air_mass)
    
    # Optical thicknesses
    tau_rayleigh = 0.0903 * (air_mass ** 0.84) * (1.0 + air_mass - air_mass ** 1.01)
    tau_aerosol = aerosol_optical_depth * air_mass
    tau_vapor = 0.077 * (water_vapor_cm * air_mass) ** 0.3
    tau_total = tau_rayleigh + tau_aerosol + tau_vapor
    
    dni = i_solar_constant * np.exp(-tau_total)
    # Global Horizontal Irradiance GHI = DNI * cos(zenith) + Diffuse
    ghi = dni * cos_z + 0.12 * i_solar_constant * (1.0 - np.exp(-tau_total)) * cos_z
    return np.maximum(0.0, ghi)


def simulate_pipeline_pressure_wave(
    length_m: float = 2000.0,
    wave_speed_c: float = 1200.0,
    nx: int = 100,
    nt: int = 80,
    leak_location_m: float = 750.0,
    leak_discharge_coeff: float = 0.02
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates water-hammer 1D transient pressure wave in an industrial pipeline
    under abrupt valve closure, with signature reflective acoustic attenuation at leak node.
    """
    x = np.linspace(0, length_m, nx)
    dx = length_m / (nx - 1)
    dt = dx / wave_speed_c  # CFL = 1.0 stability
    t = np.arange(nt) * dt
    
    p_field = np.zeros((nt, nx))
    p_steady = 50.0  # bar
    p_field[0, :] = p_steady
    
    leak_idx = int(leak_location_m / dx)
    
    # Boundary transient (valve closure at downstream boundary x=L)
    for k in range(1, nt):
        # Interior wave propagation (d2P/dt2 = c^2 d2P/dx2)
        p_field[k, 1:-1] = (2 * p_field[k-1, 1:-1] - (p_field[k-2, 1:-1] if k > 1 else p_steady) + 
                            (wave_speed_c * dt / dx)**2 * (p_field[k-1, 2:] - 2*p_field[k-1, 1:-1] + p_field[k-1, :-2]))
        
        # Upstream reservoir boundary condition (constant pressure)
        p_field[k, 0] = p_steady
        # Downstream valve transient pressure spike (Joukowsky effect)
        p_field[k, -1] = p_steady + 18.0 * np.exp(-((k * dt - 0.2) / 0.15)**2)
        
        # Leak orifice damping loss at leak_idx
        p_field[k, leak_idx] -= leak_discharge_coeff * (p_field[k, leak_idx] - 1.0)
        
    return x, t, p_field
