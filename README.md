# Data Science & Machine Learning for Physics: 30 Capstone Projects Across Industries

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Static Site: Zero-Dependency](https://img.shields.io/badge/Static%20Site-Zero--Dependency-emerald)](https://github.com/flugelsoft-01/physics-ml-book)
[![KaTeX](https://img.shields.io/badge/Math-KaTeX-blueviolet)](https://katex.org/)

A complete, self-contained, browsable static HTML monograph bridging classical physical laws (Navier-Stokes, Maxwell, Poisson, Saint-Venant, LWR) with modern Physics-Informed Machine Learning (PINNs), Fourier Neural Operators (FNO), and real-time industrial telemetry.

---

## 📖 Book Architecture & Overview

Zero heavy framework dependencies (no Quarto, Sphinx, React, or MkDocs). Built from first principles using a custom Python static site pipeline (`build_book.py`) producing high-performance HTML5, CSS3, and Vanilla JavaScript with instant client-side full-text search.

### Standardized 6-Part Capstone Anatomy
Every single project adheres to a disciplined 6-stage engineering workflow:
1. **The Physical Principle:** Governing PDEs/ODEs rendered in KaTeX, boundary & initial values, and dimensionless numbers ($Re$, $Pe$, $CFL$, $Ma$).
2. **Industrial Context & ROI:** Real-world engineering bottlenecks, operational failure modes, and quantified financial impact.
3. **Synthetic Physics Solver:** Self-contained, deterministic Python code generating physical ground-truth without broken external URLs.
4. **Physics-ML Workflow:** Invariant feature engineering, baseline ML vs. PINN/physics-constrained surrogate, and conservation-law residual checks.
5. **Interactive Visualization & Diagnostics:** Plotly interactive embeds and high-DPI publication figures.
6. **Production Deployment Notes:** Edge/cloud inference budgets, latency targets, and fail-safe safety interlocks.

---

## 🗂️ Curriculum Matrix (30 Capstone Projects)

### Part 1: Energy & Power Systems
- **Project 01:** Wind Turbine Wake Deficit Modeling (Navier-Stokes + Gaussian Process)
- **Project 02:** Smart-Grid Frequency Stability & Load Dynamics (Synchronous Generator Swing ODE)
- **Project 03:** Lithium-Ion Battery Health & Degradation (Electrochemical Impedance Spectroscopy + LSTM)
- **Project 04:** Nuclear Reactor Core Thermal Distribution (2D Poisson Conduction PINN)
- **Project 05:** Solar Photovoltaic Irradiance Forecasting (Beer-Lambert Atmospheric Radiative Attenuation)
- **Project 06:** Hydrocarbon Pipeline Acoustic Leak Localization (Water-Hammer Transient Wave PDEs)

### Part 2: Aerospace, Defense & Automotive
- **Project 07:** Supersonic Airfoil Aerodynamic Optimization (Euler Shock Expansion)
- **Project 08:** Orbital Debris Collision Avoidance & Trajectory Forecasting (Perturbed Two-Body J2 Ephemeris)
- **Project 09:** Autonomous Vehicle Perception & State Estimation (LiDAR/Radar Extended Kalman Filter)
- **Project 10:** Aviation Jet Engine Vibration Anomaly Detection (Harmonic FFT Spectral Autoencoders)
- **Project 11:** Atmospheric Ballistic Re-entry & Wind-Drift (Coriolis & Compressible Drag Monte Carlo)
- **Project 12:** Automotive Crash-Test Mesh Plastic Deformation (Continuum Plasticity Stress Tensors)

### Part 3: Semiconductors, Manufacturing & Materials
- **Project 13:** Crystal Lattice Density Functional Theory (DFT) Surrogates (Graph Neural Networks)
- **Project 14:** Semiconductor Wafer Photolithography Defect Classification (Hopkins Aerial Image CV)
- **Project 15:** 3D Heterogeneous Microchip Thermal Dissipation (Anisotropic Conduction PINN)
- **Project 16:** High-Temperature Superconductor Critical Temperature Discovery (BCS-McMillan Regression)
- **Project 17:** Additive Manufacturing Laser Melt-Pool Thermal Dynamics (Marangoni Convection)
- **Project 18:** Turbine Blade Mechanical Cyclic Fatigue Life (Paris-Erdogan Law Integration)

### Part 4: Earth Sciences, Climate & Infrastructure
- **Project 19:** Seismic Wave Arrival-Time Inversion & Epicenter Triangulation (Eikonal Travel-Time P/S Waves)
- **Project 20:** Urban Flood Inundation & Topographic Drainage (2D Shallow Water Saint-Venant Equations)
- **Project 21:** Oceanic Microplastic Lagrangian Particle Tracking (Geostrophic Gyre Advection)
- **Project 22:** High-Rise Building Seismic Resonance & Tuned Mass Damping (MDOF Dynamic Response)
- **Project 23:** Industrial Atmospheric Carbon Plume Dispersion (Gaussian Plume Inversion)
- **Project 24:** Glacier Mass-Balance & Ice-Sheet Creep Mechanics (Glen's Flow Law)

### Part 5: Finance, Healthcare & Complex Systems
- **Project 25:** Financial Market Flash-Crash Dynamics (2D Ising Spin Phase Transitions)
- **Project 26:** Cerebral Aneurysm Hemodynamic Shear Stress Prediction (Pulsatile Womersley Flow)
- **Project 27:** Accelerated Magnetic Resonance Imaging (MRI) Reconstruction (Compressed Sensing)
- **Project 28:** High-Frequency Limit Order Book Queue Dynamics (Fokker-Planck Diffusion PDEs)
- **Project 29:** Refractive Laser Eye Surgery Corneal Wavefront (Zernike Polynomial Optics)
- **Project 30:** Highway Traffic Flow Phase Transitions (Lighthill-Whitham-Richards Shockwave PDE)

---

## 📚 Foundational & Reference Guides

- **Paradigms of Physics-ML:** Comprehensive guide to Buckingham $\pi$, equivariance, PINNs, and Neural Operators (FNO & DeepONet).
- **Mathematical Notation & Dimensionless Numbers:** Master reference of vector calculus operators and dimensionless numbers ($Re, Ma, CFL, Pe, Pr, Nu, \alpha, Kn, Fr, Ma_T, Bi, Da$).
- **Industrial ROI & Speedup Benchmarking Matrix:** Empirical table comparing classical CFD/FEA vs. ML surrogate speedups and financial impact across all 30 projects.
- **Sensor Telemetry & Open Datasets Catalog:** Verified catalog of NOAA, NREL, NASA, USGS, and fastMRI telemetry repositories.
- **Cross-Disciplinary Glossary:** 120+ definitive terms across physics, engineering, and deep learning.
- **Interactive In-Browser Physics Sandbox:** Live client-side HTML5 canvas simulations of 1D viscous Burgers' shock formation and 2D Poisson heat relaxation.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/flugelsoft-01/physics-ml-book.git
cd physics-ml-book
```

### 2. Setup Dependencies
```bash
pip install -r pyproject.toml
# or
pip install numpy scipy pandas scikit-learn matplotlib plotly markdown pygments jinja2
```

### 3. Build the Static Book
```bash
python3 build_book.py
```
This will compile all 36 HTML pages, generate interactive Plotly components and high-DPI figures, and output the searchable book in `dist/`.

### 4. Browse Locally
```bash
python3 -m http.server 8000 -d dist
```
Open [http://localhost:8000](http://localhost:8000) in your web browser.

---

## 📄 License
This project is licensed under the MIT License.
