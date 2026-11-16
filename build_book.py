#!/usr/bin/env python3
"""
build_book.py - Custom Static Site Generator for:
'Data Science & Machine Learning for Physics: 30 Capstone Projects Across Industries'

Compiles Markdown chapters, preserves KaTeX equations, highlights code, generates
interactive Plotly and Matplotlib figures, compiles search_index.json, and outputs
a zero-dependency static site into dist/.
"""

import os
import re
import json
import shutil
import glob
from pathlib import Path
from typing import Dict, List, Any

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import markdown
from jinja2 import Environment, FileSystemLoader

# Ensure figures are generated for Part 1 & Flagships
try:
    import generate_part1_figures
except ImportError:
    generate_part1_figures = None

try:
    import generate_flagship_figures
except ImportError:
    generate_flagship_figures = None

# Import curriculum definitions
CURRICULUM = [
    {
        "part_id": 1,
        "name": "Part 1: Energy & Power Systems",
        "badge": "Projects 01–06",
        "folder": "part1_energy",
        "chapters": [
            {
                "num": 1,
                "slug": "01_wind_wake",
                "title": "Wind Turbine Wake Deficit Modeling & Micro-Siting",
                "short_title": "Wind Turbine Wake Modeling",
                "subtitle": "Coupling Navier-Stokes Momentum Deficit with Physics-Constrained Gaussian Processes",
                "industry": "Renewable Energy & Wind Power",
                "difficulty": "Intermediate",
                "reading_time": 25,
                "keywords": ["Navier-Stokes", "Jensen-Gaussian Wake", "Gaussian Process", "Betz Limit", "AEP Optimization"]
            },
            {
                "num": 2,
                "slug": "02_smart_grid_load",
                "title": "Smart-Grid Frequency Stability & Load Dynamics",
                "short_title": "Smart-Grid Load Dynamics",
                "subtitle": "Physics-Constrained Time-Series Forecasting via Synchronous Generator Swing ODEs",
                "industry": "Electric Utilities & Grid Operations",
                "difficulty": "Advanced",
                "reading_time": 30,
                "keywords": ["Swing Equation", "Grid Frequency", "Runge-Kutta", "Inertia Constrained", "Fault Recovery"]
            },
            {
                "num": 3,
                "slug": "03_battery_degradation",
                "title": "Lithium-Ion Battery Health & Degradation Trajectories",
                "short_title": "Battery EIS & Degradation",
                "subtitle": "Electrochemical Impedance Spectroscopy (EIS) Feature Extraction & Recurrent Neural Surrogates",
                "industry": "Automotive & Grid Storage",
                "difficulty": "Intermediate",
                "reading_time": 25,
                "keywords": ["Randles Circuit", "Warburg Diffusion", "Nyquist Plot", "State of Health", "LSTM"]
            },
            {
                "num": 4,
                "slug": "04_nuclear_core_temp",
                "title": "Nuclear Reactor Core Thermal Distribution",
                "short_title": "Nuclear Core Heat PINN",
                "subtitle": "2D Steady-State Poisson Heat Conduction via Physics-Informed Neural Networks (PINNs)",
                "industry": "Nuclear Power & Thermal Hydraulics",
                "difficulty": "Advanced",
                "reading_time": 35,
                "keywords": ["Heat Equation", "PINN", "Boundary Conditions", "Thermal Residual", "Coolant Flow"]
            },
            {
                "num": 5,
                "slug": "05_solar_irradiance",
                "title": "Solar Photovoltaic Irradiance Forecasting",
                "short_title": "Solar Irradiance Forecasting",
                "subtitle": "Atmospheric Radiative Transfer & Optical Depth Attenuation Hybrid Models",
                "industry": "Solar Utilities & Energy Trading",
                "difficulty": "Intermediate",
                "reading_time": 25,
                "keywords": ["Beer-Lambert Law", "Air Mass", "Clearsky DNI", "Atmospheric Attenuation", "Hybrid ML"]
            },
            {
                "num": 6,
                "slug": "06_pipeline_leak",
                "title": "Hydrocarbon Pipeline Acoustic Leak Localization",
                "short_title": "Pipeline Acoustic Leak Detection",
                "subtitle": "Water-Hammer Transient Wave PDEs & Inverse Acoustic Parameter Inversion",
                "industry": "Oil & Gas Infrastructure",
                "difficulty": "Advanced",
                "reading_time": 30,
                "keywords": ["Joukowsky Pressure Wave", "Water Hammer", "Inverse Problem", "Acoustic Attenuation", "Transient Solver"]
            }
        ]
    },
    {
        "part_id": 2,
        "name": "Part 2: Aerospace, Defense & Automotive",
        "badge": "Projects 07–12",
        "folder": "part2_aerospace",
        "chapters": [
            {
                "num": 7,
                "slug": "07_supersonic_airfoil",
                "title": "Supersonic Airfoil Aerodynamic Optimization",
                "short_title": "Supersonic Airfoil Optimization",
                "subtitle": "Euler Shock-Expansion Theory Coupled with Surrogate Deep Networks",
                "industry": "Aerospace & Supersonic Flight",
                "difficulty": "Advanced",
                "reading_time": 30,
                "keywords": ["Shock Expansion", "Prandtl-Meyer", "Wave Drag", "Euler Equations", "Surrogate Model"]
            },
            {
                "num": 8,
                "slug": "08_orbital_debris",
                "title": "Orbital Debris Collision Avoidance & Trajectory Forecasting",
                "short_title": "Orbital Debris Collision Avoidance",
                "subtitle": "Perturbed Two-Body Astrodynamics & Conjunction Assessment Models",
                "industry": "Space Systems & Satellite Fleet Ops",
                "difficulty": "Advanced",
                "reading_time": 30,
                "keywords": ["Keplerian Orbits", "J2 Perturbation", "Conjunction Assessment", "Ephemeris", "Physics ML"]
            },
            {
                "num": 9,
                "slug": "09_lidar_radar_fusion",
                "title": "Autonomous Vehicle Perception & Rigid-Body State Estimation",
                "short_title": "Autonomous LiDAR-Radar Fusion",
                "subtitle": "Extended Kalman Filtering & Rigid-Body Kinematic Invariants",
                "industry": "Autonomous Vehicles & Robotics",
                "difficulty": "Intermediate",
                "reading_time": 25,
                "keywords": ["Extended Kalman Filter", "Rigid Body Dynamics", "Sensor Fusion", "Kinematics", "Covariance"]
            },
            {
                "num": 10,
                "slug": "10_turbine_vibration",
                "title": "Aviation Jet Engine Vibration Anomaly Detection",
                "short_title": "Jet Engine Vibration Anomaly",
                "subtitle": "Harmonic Fast Fourier Transform Spectral Decomposition & Autoencoders",
                "industry": "Aviation & MRO Defense",
                "difficulty": "Intermediate",
                "reading_time": 25,
                "keywords": ["Harmonic FFT", "Rotor Dynamics", "Autoencoder", "Spectral Density", "Bearing Faults"]
            },
            {
                "num": 11,
                "slug": "11_ballistic_monte_carlo",
                "title": "Atmospheric Ballistic Re-entry & Wind-Drift Monte Carlo",
                "short_title": "Ballistic Wind-Drift Trajectory",
                "subtitle": "Compressible Drag, Coriolis Force, and Probabilistic Trajectory Ensembles",
                "industry": "Defense & Launch Systems",
                "difficulty": "Advanced",
                "reading_time": 30,
                "keywords": ["Coriolis Force", "Mach-Dependent Drag", "Monte Carlo", "Runge-Kutta", "Dispersion Ellipse"]
            },
            {
                "num": 12,
                "slug": "12_crash_mesh_deform",
                "title": "Automotive Crash-Test Mesh Plastic Deformation",
                "short_title": "Crash-Test Mesh Deformation",
                "subtitle": "Continuum Plasticity Stress Tensors & Geometric Graph Neural Networks",
                "industry": "Automotive Safety & Structural Engineering",
                "difficulty": "Advanced",
                "reading_time": 35,
                "keywords": ["Continuum Mechanics", "von Mises Stress", "Graph Neural Network", "Finite Element", "Plasticity"]
            }
        ]
    },
    {
        "part_id": 3,
        "name": "Part 3: Semiconductors, Manufacturing & Materials",
        "badge": "Projects 13–18",
        "folder": "part3_manufacturing",
        "chapters": [
            {
                "num": 13,
                "slug": "13_crystal_dft_surrogate",
                "title": "Crystal Lattice Density Functional Theory (DFT) Surrogates",
                "short_title": "Crystal DFT Surrogate",
                "subtitle": "Electronic Bandgap & Formation Energy Prediction via Invariant Features",
                "industry": "Advanced Semiconductor Materials",
                "difficulty": "Advanced",
                "reading_time": 30,
                "keywords": ["DFT", "Formation Energy", "Crystal Graph", "Schrodinger Approximation", "Materials Discovery"]
            },
            {
                "num": 14,
                "slug": "14_wafer_defect_cv",
                "title": "Semiconductor Wafer Photolithography Defect Classification",
                "short_title": "Wafer Photolithography Defect CV",
                "subtitle": "Fourier Optics Diffraction Pattern Recognition & Spatial Defect Signatures",
                "industry": "Semiconductor Fabrication (Fab)",
                "difficulty": "Intermediate",
                "reading_time": 25,
                "keywords": ["Photolithography", "Diffraction", "Wafer Map", "Computer Vision", "Yield Optimization"]
            },
            {
                "num": 15,
                "slug": "15_microchip_thermal",
                "title": "3D Heterogeneous Microchip Thermal Dissipation",
                "short_title": "3D Microchip Thermal Dissipation",
                "subtitle": "Multilayer Anisotropic Heat Conduction PINN with Joule Heat Generation",
                "industry": "Microprocessor Packaging & EDA",
                "difficulty": "Advanced",
                "reading_time": 35,
                "keywords": ["Anisotropic Conduction", "Joule Heating", "3D IC Packaging", "Thermal Hotspots", "PINN"]
            },
            {
                "num": 16,
                "slug": "16_superconductor_tc",
                "title": "High-Temperature Superconductor Critical Temperature Discovery",
                "short_title": "Superconductor Tc Discovery",
                "subtitle": "BCS Phonon Coupling & Supervised Physical Regression",
                "industry": "Quantum Computing & Power Transmission",
                "difficulty": "Advanced",
                "reading_time": 30,
                "keywords": ["Superconductivity", "BCS Theory", "Electron-Phonon Coupling", "Critical Temperature", "Regression"]
            },
            {
                "num": 17,
                "slug": "17_melt_pool_instability",
                "title": "Additive Manufacturing Laser Melt-Pool Thermal Dynamics",
                "short_title": "Laser Melt-Pool Instability",
                "subtitle": "Marangoni Convection, Stefan Phase Change, and Thermal Keyhole Prediction",
                "industry": "3D Metal Printing & Industrial Additive",
                "difficulty": "Advanced",
                "reading_time": 35,
                "keywords": ["Marangoni Effect", "Stefan Problem", "Keyhole Porosity", "Laser Powder Bed", "Thermal Camera"]
            },
            {
                "num": 18,
                "slug": "18_cyclic_fatigue_life",
                "title": "Turbine Blade Mechanical Cyclic Fatigue Life Estimation",
                "short_title": "Cyclic Fatigue Crack Life",
                "subtitle": "Paris-Erdogan Law Integration with Stress-Life (S-N) PINN Formulations",
                "industry": "Heavy Machinery & Structural Integrity",
                "difficulty": "Intermediate",
                "reading_time": 25,
                "keywords": ["Paris Law", "Stress Intensity Factor", "S-N Curve", "Fatigue Crack Growth", "Weibull Survival"]
            }
        ]
    },
    {
        "part_id": 4,
        "name": "Part 4: Earth Sciences, Climate & Infrastructure",
        "badge": "Projects 19–24",
        "folder": "part4_earth",
        "chapters": [
            {
                "num": 19,
                "slug": "19_earthquake_triangulation",
                "title": "Seismic Wave Arrival-Time Inversion & Epicenter Triangulation",
                "short_title": "Seismic Wave Triangulation",
                "subtitle": "P-Wave and S-Wave Elastic Refraction Physics with Inversion Neural Networks",
                "industry": "Geophysical Hazards & Seismology",
                "difficulty": "Advanced",
                "reading_time": 30,
                "keywords": ["Seismic Waves", "P-Wave S-Wave", "Eikonal Equation", "Hypocenter Inversion", "Travel Time"]
            },
            {
                "num": 20,
                "slug": "20_flood_inundation",
                "title": "Urban Flood Inundation & Topographic Drainage",
                "short_title": "Urban Flood Inundation",
                "subtitle": "2D Shallow Water Saint-Venant Equations Surrogates for Real-Time Warning",
                "industry": "Civil Protection & Climate Resilience",
                "difficulty": "Advanced",
                "reading_time": 35,
                "keywords": ["Saint-Venant Equations", "Shallow Water", "Hydrodynamic Surrogate", "Digital Elevation Model", "Floodplain"]
            },
            {
                "num": 21,
                "slug": "21_ocean_microplastics",
                "title": "Oceanic Microplastic Lagrangian Particle Tracking",
                "short_title": "Ocean Microplastic Tracking",
                "subtitle": "Hamiltonian Chaotic Advection in Oceanic Gyres & Geostrophic Flow Fields",
                "industry": "Marine Ecology & Environmental Monitoring",
                "difficulty": "Intermediate",
                "reading_time": 25,
                "keywords": ["Lagrangian Drift", "Geostrophic Flow", "Subtropical Gyres", "Chaotic Advection", "Particle Tracking"]
            },
            {
                "num": 22,
                "slug": "22_building_seismic_damping",
                "title": "High-Rise Building Seismic Resonance & Tuned Mass Damping",
                "short_title": "Building Seismic Resonance",
                "subtitle": "Coupled Multi-Degree-of-Freedom Harmonic Oscillators & Active Control Models",
                "industry": "Civil Structural Engineering",
                "difficulty": "Advanced",
                "reading_time": 30,
                "keywords": ["Tuned Mass Damper", "Eigenfrequencies", "MDOF Dynamic Response", "Ground Acceleration", "State-Space"]
            },
            {
                "num": 23,
                "slug": "23_atmospheric_carbon",
                "title": "Industrial Atmospheric Carbon Plume Dispersion",
                "short_title": "Atmospheric Carbon Plume",
                "subtitle": "Gaussian Plume Advection-Diffusion Equations & Sensor Array Inversion",
                "industry": "Environmental Compliance & Emissions Monitoring",
                "difficulty": "Intermediate",
                "reading_time": 25,
                "keywords": ["Gaussian Plume", "Advection-Diffusion", "Atmospheric Stability Classes", "Source Inversion", "Sensor Array"]
            },
            {
                "num": 24,
                "slug": "24_glacier_mass_balance",
                "title": "Glacier Mass-Balance & Ice-Sheet Creep Mechanics",
                "short_title": "Glacier Creep & Mass Balance",
                "subtitle": "Glen's Non-Newtonian Flow Law & Satellite Altimetry Physics-ML",
                "industry": "Cryospheric Science & Climate Assessment",
                "difficulty": "Advanced",
                "reading_time": 30,
                "keywords": ["Glen Flow Law", "Ice Dynamics", "Non-Newtonian Fluid", "Satellite Altimetry", "Mass Balance"]
            }
        ]
    },
    {
        "part_id": 5,
        "name": "Part 5: Finance, Healthcare & Complex Systems",
        "badge": "Projects 25–30",
        "folder": "part5_finance_health",
        "chapters": [
            {
                "num": 25,
                "slug": "25_market_volatility_ising",
                "title": "Financial Market Flash-Crash Dynamics via 2D Ising Spin Systems",
                "short_title": "Market Volatility Ising Models",
                "subtitle": "Statistical Mechanics of Phase Transitions, Ferromagnetic Coupling, and Liquidity Cascades",
                "industry": "Quantitative Finance & Algorithmic Trading",
                "difficulty": "Advanced",
                "reading_time": 30,
                "keywords": ["Ising Model", "Phase Transition", "Critical Opalescence", "Order Parameter", "Flash Crash"]
            },
            {
                "num": 26,
                "slug": "26_aneurysm_blood_flow",
                "title": "Cerebral Aneurysm Hemodynamic Shear Stress Prediction",
                "short_title": "Cerebral Aneurysm Hemodynamics",
                "subtitle": "Pulsatile Navier-Stokes & Womersley Velocity Profile Surrogates",
                "industry": "Cardiovascular Medicine & Biomedical Tech",
                "difficulty": "Advanced",
                "reading_time": 35,
                "keywords": ["Pulsatile Blood Flow", "Wall Shear Stress", "Womersley Number", "Navier-Stokes", "Rupture Risk"]
            },
            {
                "num": 27,
                "slug": "27_accelerated_mri_cs",
                "title": "Accelerated Magnetic Resonance Imaging (MRI) Reconstruction",
                "short_title": "Accelerated MRI Compressed Sensing",
                "subtitle": "Fourier K-Space Subsampling & Compressed Sensing Deep Regularization",
                "industry": "Diagnostic Imaging & Medical Devices",
                "difficulty": "Advanced",
                "reading_time": 35,
                "keywords": ["K-Space", "Compressed Sensing", "Fourier Transform", "Total Variation", "Deep Reconstruction"]
            },
            {
                "num": 28,
                "slug": "28_hft_queue_mechanics",
                "title": "High-Frequency Limit Order Book Queue Dynamics",
                "short_title": "Order Book Queue Mechanics",
                "subtitle": "Fokker-Planck Diffusion PDEs & Microstructure Particle Surrogates",
                "industry": "Market Microstructure & FinTech",
                "difficulty": "Advanced",
                "reading_time": 30,
                "keywords": ["Fokker-Planck", "Master Equation", "Limit Order Book", "Queueing Theory", "Microstructure"]
            },
            {
                "num": 29,
                "slug": "29_corneal_wavefront",
                "title": "Refractive Laser Eye Surgery Corneal Wavefront Reconstruction",
                "short_title": "Corneal Optical Wavefront",
                "subtitle": "Zernike Polynomial Wavefront Aberration Expansion & Ray-Tracing Optics",
                "industry": "Ophthalmic Surgery & Laser Optics",
                "difficulty": "Intermediate",
                "reading_time": 25,
                "keywords": ["Zernike Polynomials", "Wavefront Aberration", "Optical Ray Tracing", "LASIK Ablation", "Point Spread Function"]
            },
            {
                "num": 30,
                "slug": "30_traffic_phase_lwr",
                "title": "Highway Traffic Flow Phase Transitions & Shockwave Dynamics",
                "short_title": "Highway Traffic Phase Transitions",
                "subtitle": "Lighthill-Whitham-Richards (LWR) Hyperbolic Conservation PDEs",
                "industry": "Intelligent Transportation & Smart Cities",
                "difficulty": "Intermediate",
                "reading_time": 25,
                "keywords": ["LWR Model", "Hyperbolic PDE", "Shockwave", "Fundamental Diagram", "Traffic Congestion"]
            }
        ]
    }
]

SUPPLEMENTARY_PAGES = [
    {
        "slug": "foundations",
        "title": "Foundations & Paradigms of Physics-Informed ML",
        "short_title": "Paradigms of Physics-ML",
        "subtitle": "Buckingham Pi, Equivariance, PINNs, and Neural Operators (FNO & DeepONet)",
        "badge": "Theory",
        "industry": "Theoretical Physics & Foundations",
        "difficulty": "Foundational",
        "reading_time": 20,
        "keywords": ["Buckingham Pi", "PINN", "Fourier Neural Operator", "DeepONet", "Inductive Bias", "Equivariance"]
    },
    {
        "slug": "notation",
        "title": "Mathematical Notation & Dimensionless Numbers",
        "short_title": "Notation & Dimensionless Numbers",
        "subtitle": "Vector Calculus Operators, Differential Invariants, and Master Dimensionless Ratios",
        "badge": "Reference",
        "industry": "Applied Mathematics & Mechanics",
        "difficulty": "Foundational",
        "reading_time": 15,
        "keywords": ["Reynolds", "Mach", "Peclet", "Laplacian", "Divergence", "Womersley", "Dimensionless Numbers"]
    },
    {
        "slug": "benchmarks",
        "title": "Industrial ROI, Latency & Speedup Benchmarking Matrix",
        "short_title": "Industrial ROI Benchmarks",
        "subtitle": "Empirical Comparison of 30 Projects: Classical CFD/FEA vs Physics-ML Surrogates",
        "badge": "Benchmarks",
        "industry": "Cross-Industry Systems Engineering",
        "difficulty": "Executive / Architecture",
        "reading_time": 15,
        "keywords": ["ROI", "Speedup", "Inference Latency", "Edge Hardware", "Financial Impact"]
    },
    {
        "slug": "datasets",
        "title": "Sensor Telemetry & Open Physical Benchmark Datasets",
        "short_title": "Open Datasets Catalog",
        "subtitle": "Verified Catalog of NOAA, NREL, NASA, USGS, and fastMRI Telemetry Repositories",
        "badge": "Data Catalog",
        "industry": "Data Engineering & Remote Sensing",
        "difficulty": "Practical Guide",
        "reading_time": 15,
        "keywords": ["NREL", "NASA C-MAPSS", "fastMRI", "USGS Seismic", "Copernicus CMEMS", "LOBSTER"]
    },
    {
        "slug": "glossary",
        "title": "Cross-Disciplinary Physics-ML Glossary",
        "short_title": "Physics-ML Glossary",
        "subtitle": "120+ Definitive Terms Across Continuum Mechanics, Statistical Physics, and Deep Learning",
        "badge": "Glossary",
        "industry": "Physics & Computer Science Lexicon",
        "difficulty": "Reference",
        "reading_time": 25,
        "keywords": ["Glossary", "Continuum Mechanics", "Navier-Stokes", "Ising", "Autograd", "Collocation"]
    },
    {
        "slug": "playground",
        "title": "Interactive Physics-ML In-Browser Sandbox",
        "short_title": "Live Interactive Sandbox",
        "subtitle": "Real-Time 1D Viscous Burgers' Shock Formation & 2D Heat Diffusion Solver",
        "badge": "Interactive Lab",
        "industry": "Client-Side Interactive Simulation",
        "difficulty": "Interactive",
        "reading_time": 15,
        "keywords": ["Burgers Equation", "Heat Equation", "Interactive Canvas", "Shock Formation", "Lax-Friedrichs"]
    }
]


def extract_math_blocks(text: str):
    """
    Safely preserves KaTeX math ($...$ and $$...$$) from being corrupted
    by Python-Markdown's emphasis/underscore processing.
    """
    math_map = {}
    counter = 0

    # 1. Block math: $$ ... $$
    def block_repl(match):
        nonlocal counter
        key = f"@@@MATH_BLOCK_{counter}@@@"
        math_map[key] = f'<div class="katex-display">$${match.group(1)}$$</div>'
        counter += 1
        return f"\n\n{key}\n\n"

    text = re.sub(r"\$\$(.*?)\$\$", block_repl, text, flags=re.DOTALL)

    # 2. Inline math: $ ... $ (ignoring currency like $100 or shell $VAR)
    def inline_repl(match):
        nonlocal counter
        key = f"@@@MATH_INLINE_{counter}@@@"
        math_map[key] = f"${match.group(1)}$"
        counter += 1
        return key

    text = re.sub(r"(?<!\\)\$(?!\s)([^\$\n]+?)(?<!\s)\$", inline_repl, text)
    return text, math_map


def restore_math_blocks(html: str, math_map: Dict[str, str]) -> str:
    """Restores saved KaTeX math blocks after markdown compilation."""
    for key, val in math_map.items():
        # Match standalone paragraph tags surrounding block math
        html = html.replace(f"<p>{key}</p>", val)
        html = html.replace(key, val)
    return html


def convert_callouts(html: str) -> str:
    """Transforms blockquotes with [!NOTE], [!TIP], etc., into styled callouts."""
    callout_pattern = re.compile(
        r"<blockquote>\s*<p>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|PHYSICS)\]\s*(.*?)</p>\s*(.*?)</blockquote>",
        re.DOTALL | re.IGNORECASE
    )

    def repl(m):
        type_str = m.group(1).upper()
        header = m.group(2)
        body = m.group(3)

        cls = "callout-info"
        icon = "ℹ️"
        title = "Note"
        if type_str in ["TIP", "PHYSICS"]:
            cls = "callout-physics"
            icon = "⚛️"
            title = "Physics Principle & Governing Constraint"
        elif type_str in ["IMPORTANT", "WARNING"]:
            cls = "callout-warning"
            icon = "⚠️"
            title = "Engineering Warning"

        return f'''<div class="callout {cls}">
          <div class="callout-title"><span>{icon}</span> <span>{title}: {header}</span></div>
          {f"<p>{body}</p>" if body.strip() else ""}
        </div>'''

    return callout_pattern.sub(repl, html)


def build():
    root_dir = Path(__file__).resolve().parent
    dist_dir = root_dir / "dist"
    templates_dir = root_dir / "templates"
    assets_dir = root_dir / "assets"
    chapters_dir = root_dir / "chapters"
    figures_dir = dist_dir / "assets" / "figures"

    # Setup directories
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(dist_dir / "chapters", exist_ok=True)
    os.makedirs(dist_dir / "assets", exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Generate Figures and Interactive Embeds
    os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
    if generate_part1_figures:
        generate_part1_figures.run()
    if generate_flagship_figures:
        generate_flagship_figures.run()

    # Copy static assets (css, js, icons)
    for asset_file in assets_dir.glob("*"):
        if asset_file.is_file():
            shutil.copy2(asset_file, dist_dir / "assets" / asset_file.name)

    # Cloudflare Pages specific deployment files (_headers, _redirects)
    for cf_file in ["_headers", "_redirects"]:
        src_cf = assets_dir / cf_file
        if src_cf.exists():
            shutil.copy2(src_cf, dist_dir / cf_file)

    # Copy shared_utils for Pyodide in-browser runtime
    dist_su = dist_dir / "shared_utils"
    dist_su.mkdir(parents=True, exist_ok=True)
    for su_file in (root_dir / "shared_utils").glob("*.py"):
        shutil.copy2(su_file, dist_su / su_file.name)

    # Generate 30 Jupyter Notebooks (.ipynb)
    try:
        from generate_notebooks import generate_all_notebooks
        generate_all_notebooks()
    except Exception as e:
        print(f"Warning: Could not generate notebooks: {e}")

    # Jinja2 environment
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    book_template = env.get_template("book_layout.html")
    index_template = env.get_template("index_layout.html")

    # Flatten chapters for navigation (prev/next)
    flat_chapters = []
    for part in CURRICULUM:
        for ch in part["chapters"]:
            ch_copy = dict(ch)
            ch_copy["part_name"] = part["name"]
            ch_copy["number_str"] = f"{ch['num']:02d}"
            flat_chapters.append(ch_copy)

    search_index = []

    print(f"Compiling {len(flat_chapters)} chapters...")

    for i, ch in enumerate(flat_chapters):
        slug = ch["slug"]
        part_folder = [p["folder"] for p in CURRICULUM if ch["num"] in [c["num"] for c in p["chapters"]]][0]
        md_path = chapters_dir / part_folder / f"{slug}.md"

        if not md_path.exists():
            print(f"Warning: Markdown source not found: {md_path}")
            continue

        raw_md = md_path.read_text(encoding="utf-8")

        # Extract title from markdown if present (# Title)
        first_h1 = re.search(r"^#\s+(.+)$", raw_md, re.MULTILINE)
        if first_h1:
            raw_title = first_h1.group(1).strip()
            # Remove leading title from md content to avoid duplicate H1
            raw_md = re.sub(r"^#\s+.+$", "", raw_md, count=1, flags=re.MULTILINE).strip()

        # Handle math placeholders
        processed_md, math_map = extract_math_blocks(raw_md)

        # Markdown to HTML
        md_parser = markdown.Markdown(extensions=[
            "fenced_code",
            "tables",
            "toc",
            "attr_list",
            "def_list",
            "codehilite"
        ])
        content_html = md_parser.convert(processed_md)
        content_html = restore_math_blocks(content_html, math_map)
        content_html = convert_callouts(content_html)

        # Build search index entry
        plain_text = re.sub(r"<[^>]+>", " ", content_html)
        plain_text = re.sub(r"\s+", " ", plain_text).strip()
        search_index.append({
            "url": f"chapters/{slug}.html",
            "slug": slug,
            "title": f"Project {ch['number_str']}: {ch['title']}",
            "part": ch["part_name"],
            "keywords": ch.get("keywords", []),
            "content": plain_text[:3500]  # Store first 3.5k chars for quick client matching
        })

        # Prev and next navigation
        prev_ch = flat_chapters[i - 1] if i > 0 else None
        next_ch = flat_chapters[i + 1] if i < len(flat_chapters) - 1 else None

        ch_ctx = dict(ch)
        ch_ctx["content_html"] = content_html
        ch_ctx["is_capstone"] = True
        ch_ctx["colab_url"] = f"https://colab.research.google.com/github/flugelsoft-01/physics-ml-book/blob/main/notebooks/{slug}.ipynb"
        ch_ctx["notebook_url"] = f"../notebooks/{slug}.ipynb"

        rendered_html = book_template.render(
            curriculum=CURRICULUM,
            supplementary=SUPPLEMENTARY_PAGES,
            chapter=ch_ctx,
            prev_chapter=prev_ch,
            next_chapter=next_ch,
            base_path="../"
        )

        out_path = dist_dir / "chapters" / f"{slug}.html"
        out_path.write_text(rendered_html, encoding="utf-8")

    # Compile Supplementary Reference Pages
    print(f"Compiling {len(SUPPLEMENTARY_PAGES)} supplementary reference pages...")
    for j, sp in enumerate(SUPPLEMENTARY_PAGES):
        slug = sp["slug"]
        md_path = chapters_dir / "supplementary" / f"{slug}.md"
        if not md_path.exists():
            print(f"Warning: Supplementary markdown not found: {md_path}")
            continue

        raw_md = md_path.read_text(encoding="utf-8")
        raw_md = re.sub(r"^#\s+.+$", "", raw_md, count=1, flags=re.MULTILINE).strip()

        processed_md, math_map = extract_math_blocks(raw_md)
        md_parser = markdown.Markdown(extensions=[
            "fenced_code", "tables", "toc", "attr_list", "def_list", "codehilite"
        ])
        content_html = md_parser.convert(processed_md)
        content_html = restore_math_blocks(content_html, math_map)
        content_html = convert_callouts(content_html)

        # Index entry
        plain_text = re.sub(r"<[^>]+>", " ", content_html)
        plain_text = re.sub(r"\s+", " ", plain_text).strip()
        search_index.append({
            "url": f"chapters/{slug}.html",
            "slug": slug,
            "title": sp["title"],
            "part": "Foundations & Reference",
            "keywords": sp.get("keywords", []),
            "content": plain_text[:3500]
        })

        sp_ctx = dict(sp)
        sp_ctx["content_html"] = content_html
        sp_ctx["part_name"] = "Foundations & Reference"
        sp_ctx["number_str"] = sp["badge"]
        sp_ctx["is_capstone"] = False
        sp_ctx["colab_url"] = None
        sp_ctx["notebook_url"] = None

        prev_p = SUPPLEMENTARY_PAGES[j - 1] if j > 0 else None
        next_p = SUPPLEMENTARY_PAGES[j + 1] if j < len(SUPPLEMENTARY_PAGES) - 1 else None

        rendered_html = book_template.render(
            curriculum=CURRICULUM,
            supplementary=SUPPLEMENTARY_PAGES,
            chapter=sp_ctx,
            prev_chapter=prev_p,
            next_chapter=next_p,
            base_path="../"
        )
        (dist_dir / "chapters" / f"{slug}.html").write_text(rendered_html, encoding="utf-8")

    # Render index.html
    index_html = index_template.render(
        curriculum=CURRICULUM,
        supplementary=SUPPLEMENTARY_PAGES,
        base_path="./"
    )
    (dist_dir / "index.html").write_text(index_html, encoding="utf-8")

    # Write search_index.json
    (dist_dir / "search_index.json").write_text(json.dumps(search_index, indent=2), encoding="utf-8")

    # Generate XML Sitemap (SEO)
    base_url = "https://physics-ml-book.org"
    sitemap_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
        f'  <url><loc>{base_url}/index.html</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>'
    ]
    for item in search_index:
        priority = "0.9" if "Foundations" in item["part"] else "0.8"
        sitemap_lines.append(f'  <url><loc>{base_url}/{item["url"]}</loc><changefreq>monthly</changefreq><priority>{priority}</priority></url>')
    sitemap_lines.append('</urlset>')
    (dist_dir / "sitemap.xml").write_text("\n".join(sitemap_lines), encoding="utf-8")

    # Generate robots.txt
    robots_content = f"""User-agent: *
Allow: /

Sitemap: {base_url}/sitemap.xml
"""
    (dist_dir / "robots.txt").write_text(robots_content, encoding="utf-8")

    print(f"Successfully generated book in {dist_dir}!")
    print(f"Total chapters: {len(search_index)}")
    print(f"Search index entries: {len(search_index)}")
    print(f"Generated SEO sitemap.xml with {len(search_index) + 1} URLs")
    print(f"Generated robots.txt")


if __name__ == "__main__":
    build()
