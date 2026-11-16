# Cross-Disciplinary Physics-ML Glossary

An authoritative reference defining key terminology bridging classical continuum mechanics, theoretical physics, dynamic systems, and modern deep learning.

---

### A

- **Actuator Disk Model:** An analytical aerodynamic representation of a wind turbine or propeller rotor replacing individual blades with a permeable disk that exerts an axial thrust force deficit on passing fluid.
- **Advection:** The transport of a physical substance, scalar quantity, or conserved fluid property solely due to the bulk velocity field of the carrying fluid ($(\mathbf{u} \cdot \nabla)\phi$).
- **Arrhenius Equation:** An exponential formula modeling the temperature dependence of chemical reaction and thermal degradation rates: $k(T) = A \exp(-E_a / RT)$.
- **Automatic Differentiation (Autograd):** A chain-rule graph execution technique that computes exact mathematical derivatives of arbitrary computational functions up to machine precision without numerical finite-difference discretization artifacts.

### B

- **BCS Theory (Bardeen-Cooper-Schrieffer):** The microscopic theory of superconductivity explaining zero electrical resistance below critical temperature $T_c$ via phonon-mediated electron pairing into Cooper pairs.
- **Beer-Lambert-Bouguer Law:** The exponential attenuation of radiant electromagnetic flux propagating through an absorbing or scattering medium: $I = I_0 \exp(-\tau \cdot m)$.
- **Betz Limit:** The theoretical upper bound stating that no horizontal-axis wind turbine can capture more than $16/27 \approx 59.3\%$ of the kinetic energy in oncoming wind.
- **Buckingham $\pi$ Theorem:** A fundamental dimensional analysis theorem stating that a physically invariant relationship among $n$ variables with $k$ physical dimensions can be reformulated as a function of $n - k$ dimensionless parameters.

### C

- **Collocation Points:** Unlabeled spatial and temporal coordinates $(\mathbf{x}_i, t_i)$ distributed across a domain interior where neural network outputs are differentiated to evaluate differential equation residual losses.
- **Continuity Equation:** The mathematical statement of conservation of mass in a continuum: $\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0$.
- **Coriolis Force:** An apparent inertial force in a rotating planetary reference frame acting perpendicular to the velocity vector: $\mathbf{a}_{\text{Coriolis}} = -2(\boldsymbol{\Omega} \times \mathbf{v})$.
- **Courant-Friedrichs-Lewy (CFL) Condition:** A necessary numerical stability constraint for explicit hyperbolic PDE integration requiring that numerical domain of dependence covers the physical domain: $CFL = \frac{v \Delta t}{\Delta x} \le 1.0$.

### D

- **DeepONet (Deep Operator Network):** A dual-network architecture (Branch and Trunk) learning non-linear operators mapping continuous function spaces to continuous function outputs based on the universal operator approximation theorem.
- **Departure from Nucleate Boiling (DNB):** The critical thermal threshold in a liquid-cooled nuclear reactor core where vapor bubbles form an insulating film over fuel rods, drastically reducing convective heat transfer and risking clad meltdown.
- **Deviatoric Stress Tensor:** The trace-free component of the stress tensor ($\mathbf{s} = \boldsymbol{\sigma} - \frac{1}{3}\text{Tr}(\boldsymbol{\sigma})\mathbf{I}$) responsible for shearing and plastic shape distortion without changing volume.

### E

- **Eikonal Equation:** The high-frequency geometric optics and acoustics approximation of the wave equation describing the spatial propagation of travel-time wavefronts: $|\nabla T(\mathbf{x})|^2 = 1 / v(\mathbf{x})^2$.
- **Electrochemical Impedance Spectroscopy (EIS):** An analytical diagnostic measuring complex frequency-dependent electrical impedance $Z(\omega)$ by applying a small sinusoidal voltage perturbation across a battery or fuel cell.
- **Equivariance:** A structural property of an operator where applying a spatial transformation (e.g. rotation $\mathbf{R} \in SO(3)$) to input features results in an identically transformed output: $\mathbf{f}(\mathbf{R}\mathbf{x}) = \mathbf{R}\mathbf{f}(\mathbf{x})$.
- **Euler Equations:** The non-linear hyperbolic conservation laws governing inviscid, non-heat-conducting compressible fluid dynamics.

### F

- **Fast Fourier Transform (FFT):** An algorithm computing the Discrete Fourier Transform of an $N$-element sequence in $\mathcal{O}(N \log N)$ complexity, transforming spatial/temporal signals into frequency-domain spectra.
- **Fokker-Planck Equation:** A partial differential equation describing the time evolution of the continuous probability density function of particles subject to deterministic drag drift and stochastic Brownian diffusion forces.
- **Fourier Neural Operator (FNO):** A deep learning operator architecture that parameterizes integral kernel operators directly in Fourier frequency space via FFT and inverse FFT layers.

### G

- **Galilean Invariance:** The physical principle stating that the fundamental laws of motion are identical in all non-accelerating inertial reference frames.
- **Glen's Flow Law:** The constitutive empirical equation describing glacial ice deformation as a non-Newtonian power-law fluid: $\dot{\varepsilon} = A \tau^n$, with typical exponent $n \approx 3$.
- **Greenshields Model:** An empirical macroscopic traffic flow formulation assuming a linear decline in vehicular speed with increasing vehicular density: $v(\rho) = v_{\text{free}}(1 - \rho / \rho_{\text{jam}})$.

### H

- **Hamiltonian Mechanics:** A reformulation of classical mechanics where the state of a physical system is expressed in canonical position $\mathbf{q}$ and momentum $\mathbf{p}$ phase space coordinates, with total energy $\mathcal{H}(\mathbf{q}, \mathbf{p})$ conserved.
- **Harmonic Oscillator:** A dynamical system that, when displaced from equilibrium, experiences a restoring force proportional to displacement: $m \ddot{x} + c \dot{x} + k x = 0$.
- **Hopkins Diffraction Model:** The standard optical formulation calculating aerial image light intensity formed by partially coherent illumination through a photolithography mask.

### I

- **Inductive Bias:** The set of structural assumptions, physical symmetries, or conservation constraints encoded directly into a machine learning model architecture to prioritize plausible physical hypotheses.
- **Ising Model:** A mathematical model in statistical mechanics describing ferromagnetic phase transitions via a lattice of interacting discrete magnetic dipole spins $s_i \in \{+1, -1\}$.

### J

- **Joukowsky Equation:** The fundamental acoustic wave equation calculating the maximum instantaneous transient pressure rise resulting from an abrupt liquid valve shutdown: $\Delta P_{\text{max}} = \rho \cdot a \cdot \Delta v$.

### K

- **$k$-Space:** The spatial frequency 2D/3D Fourier domain where raw Magnetic Resonance Imaging (MRI) gradient echo signals are collected prior to inverse Fourier reconstruction.
- **Kohn-Sham Equations:** The single-particle Schrödinger-like differential equations whose ground-state solutions yield the exact electronic charge density and energy of a many-body quantum atomic system.

### L

- **Lagrangian Particle Tracking:** A fluid mechanics tracking framework following the individual discrete trajectories of particles or pollutants moving within a continuum velocity field: $\frac{d\mathbf{x}_p}{dt} = \mathbf{u}(\mathbf{x}_p, t)$.
- **Lighthill-Whitham-Richards (LWR) Model:** The fundamental macroscopic first-order kinematic wave PDE modeling traffic flow density conservation: $\frac{\partial \rho}{\partial t} + \frac{\partial (\rho v)}{\partial x} = 0$.

### M

- **Marangoni Effect:** Thermocapillary fluid convection driven by spatial gradients in surface tension along a liquid-gas or liquid-liquid interface: $\tau = \frac{d\gamma}{dT} \nabla_s T$.
- **Master Equation:** A set of coupled differential equations describing the time-dependent probability distributions of discrete states undergoing stochastic transition jump rates.
- **Munnerlyn Formula:** The geometric equation calculating corneal tissue ablation depth required to reshape refractive optical curvature during laser eye surgery.

### N

- **Navier-Stokes Equations:** The non-linear partial differential equations describing the conservation of linear momentum and mass for viscous Newtonian fluids.
- **Nyquist Plot:** A complex frequency-response locus plotting negative imaginary impedance $-Z''(\omega)$ against real impedance $Z'(\omega)$ across alternating excitation frequencies.

### P

- **Paris-Erdogan Law:** The power-law fracture mechanics relationship relating cyclic mechanical fatigue crack growth rate to stress intensity factor range: $\frac{da}{dN} = C (\Delta K)^m$.
- **Physics-Informed Neural Network (PINN):** A neural network trained by minimizing a multi-objective loss combining measured sensor data with differential equation residuals evaluated via automatic differentiation.
- **Prandtl-Meyer Expansion:** A continuous, isentropic expansion fan centered at a convex corner in supersonic flow that lowers static pressure and accelerates Mach number.

### R

- **Randles Circuit:** An equivalent electrical circuit modeling battery electrochemical cells using an electrolyte resistance, double-layer capacitance, charge-transfer resistance, and Warburg diffusion impedance.
- **Rankine-Hugoniot Conditions:** The mathematical relations derived from conservation laws describing the discontinuous jump in pressure, density, and velocity across a physical shockwave front.
- **Robin Boundary Condition:** A mixed boundary constraint specifying a linear combination of a field variable and its normal spatial derivative: $a u + b \frac{\partial u}{\partial n} = g$.
- **Rosenthal Solution:** The analytical 3D thermal conduction solution for a moving point heat source traversing a semi-infinite solid, widely applied in welding and laser 3D printing.

### S

- **Saint-Venant Equations:** The 2D hyperbolic shallow water equations derived from depth-integrating Navier-Stokes under the assumption that vertical scale is negligible relative to horizontal scale.
- **Solid Electrolyte Interphase (SEI):** A passivation layer that forms on the graphite anode surface of a lithium-ion battery due to electrolyte decomposition; its continual parabolic thickening is the primary driver of capacity fade.
- **Spectral Bias:** The mathematical tendency of gradient-based neural networks to fit smooth, low-frequency functions rapidly while taking disproportionately long to learn high-frequency physical oscillations.
- **Swing Equation:** The non-linear second-order differential equation describing the rotational electromechanical dynamics of a synchronous power generator relative to grid reference frequency: $M \ddot{\delta} + D \dot{\delta} = P_m - P_e$.

### T

- **Tuned Mass Damper (TMD):** A passive or active secondary vibrating mass mounted inside tall structures or bridges tuned to the structure's primary natural resonance frequency to dissipate seismic or aerodynamic wind vibrations.

### V

- **von Mises Stress:** An equivalent scalar stress criterion based on the second deviatoric stress invariant ($J_2$) predicting whether a ductile metallic material will undergo irreversible plastic yield.

### W

- **Warburg Impedance:** An electrical impedance element with constant $45^\circ$ phase angle modeling semi-infinite linear diffusion of ions through solid-state electrodes in electrochemistry: $Z_W = \frac{\sigma}{\sqrt{\omega}}(1 - j)$.
- **Water-Hammer:** A destructive pressure surge or acoustic shockwave caused when a fluid in motion is abruptly forced to stop or change direction inside a closed pipeline.
- **Womersley Number ($\alpha$):** A dimensionless parameter in biofluid mechanics expressing the ratio of transient pulsatile inertial forces to viscous diffusion forces in blood flow: $\alpha = R \sqrt{\frac{\omega}{\nu}}$.

### Z

- **Zernike Polynomials:** A sequence of orthogonal polynomials defined on the unit circular disk widely used in optical engineering to decompose and quantify wavefront aberrations in the human cornea and telescope lenses.
