# Mathematical Notation, Differential Operators & Dimensionless Physics Numbers

## 1. Vector Calculus & Coordinate Differential Operators

### Cartesian Coordinates $(x, y, z)$

For a scalar field $\phi(x, y, z)$ and vector field $\mathbf{u} = u_x \hat{\mathbf{i}} + u_y \hat{\mathbf{j}} + u_z \hat{\mathbf{k}}$:

| Operation | Mathematical Definition | Physical Meaning |
| :--- | :--- | :--- |
| **Gradient** $\nabla \phi$ | $\nabla \phi = \frac{\partial \phi}{\partial x}\hat{\mathbf{i}} + \frac{\partial \phi}{\partial y}\hat{\mathbf{j}} + \frac{\partial \phi}{\partial z}\hat{\mathbf{k}}$ | Direction and rate of steepest spatial ascent |
| **Divergence** $\nabla \cdot \mathbf{u}$ | $\nabla \cdot \mathbf{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} + \frac{\partial u_z}{\partial z}$ | Local volumetric source ($>0$) or sink ($<0$) density |
| **Curl** $\nabla \times \mathbf{u}$ | $\nabla \times \mathbf{u} = \begin{vmatrix} \hat{\mathbf{i}} & \hat{\mathbf{j}} & \hat{\mathbf{k}} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ u_x & u_y & u_z \end{vmatrix}$ | Local microscopic rotational vorticity vector |
| **Laplacian** $\nabla^2 \phi$ | $\nabla^2 \phi = \frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2} + \frac{\partial^2 \phi}{\partial z^2}$ | Diffusion curvature and flux divergence |
| **Material Derivative** $\frac{D}{Dt}$ | $\frac{D\mathbf{u}}{Dt} = \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u}$ | Total rate of change following a moving fluid particle |

---

### Cylindrical Coordinates $(r, \theta, z)$

$$\nabla \phi = \frac{\partial \phi}{\partial r}\hat{\mathbf{r}} + \frac{1}{r}\frac{\partial \phi}{\partial \theta}\hat{\boldsymbol{\theta}} + \frac{\partial \phi}{\partial z}\hat{\mathbf{z}}$$

$$\nabla \cdot \mathbf{u} = \frac{1}{r}\frac{\partial (r u_r)}{\partial r} + \frac{1}{r}\frac{\partial u_\theta}{\partial \theta} + \frac{\partial u_z}{\partial z}$$

$$\nabla^2 \phi = \frac{1}{r}\frac{\partial}{\partial r}\left( r \frac{\partial \phi}{\partial r} \right) + \frac{1}{r^2}\frac{\partial^2 \phi}{\partial \theta^2} + \frac{\partial^2 \phi}{\partial z^2}$$

---

## 2. Master Table of Dimensionless Numbers in Physics & Engineering

Dimensionless numbers quantify the relative magnitudes of competing physical forces (inertial, viscous, thermal, electromagnetic) in governing equations. They serve as primary invariant features in physics-informed feature pipelines:

| Number | Symbol | Definition Formula | Ratio of Physical Forces / Rates | Key Industrial Domain |
| :--- | :---: | :--- | :--- | :--- |
| **Reynolds** | $Re$ | $Re = \frac{\rho v L}{\mu} = \frac{v L}{\nu}$ | $\frac{\text{Inertial Forces}}{\text{Viscous Forces}}$ | Fluid dynamics, aerodynamics, piping |
| **Mach** | $Ma$ | $Ma = \frac{v}{a} = \frac{v}{\sqrt{\gamma R T}}$ | $\frac{\text{Flow Velocity}}{\text{Speed of Sound}}$ | Supersonic aircraft, turbine expansion |
| **Courant-Friedrichs-Lewy** | $CFL$ | $CFL = \frac{v \Delta t}{\Delta x}$ | $\frac{\text{Advective Distance per Step}}{\text{Grid Spacing}}$ | Hyperbolic PDE numerical stability |
| **Péclet** | $Pe$ | $Pe = \frac{v L}{\alpha} = Re \cdot Pr$ | $\frac{\text{Advective Transport}}{\text{Diffusive Transport}}$ | Mass transfer, thermal mixing |
| **Prandtl** | $Pr$ | $Pr = \frac{\nu}{\alpha} = \frac{c_p \mu}{k}$ | $\frac{\text{Momentum Diffusivity}}{\text{Thermal Diffusivity}}$ | Boundary layer heat transfer |
| **Nusselt** | $Nu$ | $Nu = \frac{h L}{k}$ | $\frac{\text{Convective Heat Transfer}}{\text{Conductive Heat Transfer}}$ | Nuclear cooling, microchip heat sinks |
| **Womersley** | $\alpha$ | $\alpha = R \sqrt{\frac{\omega \rho}{\mu}} = R \sqrt{\frac{\omega}{\nu}}$ | $\frac{\text{Transient Inertial Frequency}}{\text{Viscous Diffusion Forces}}$ | Arterial hemodynamics, pulsatile flow |
| **Knudsen** | $Kn$ | $Kn = \frac{\lambda_{\text{mfp}}}{L}$ | $\frac{\text{Molecular Mean Free Path}}{\text{Characteristic Length}}$ | Rarefied aerodynamics, MEMS vacuum |
| **Froude** | $Fr$ | $Fr = \frac{v}{\sqrt{g L}}$ | $\frac{\text{Inertial Forces}}{\text{Gravitational Forces}}$ | Ocean waves, shallow water, open channel |
| **Marangoni** | $Ma_T$| $Ma_T = -\frac{d\gamma}{dT} \frac{\Delta T L}{\mu \alpha}$ | $\frac{\text{Surface Tension Gradient Forces}}{\text{Viscous Dissipation}}$ | Laser additive manufacturing, weld pools |
| **Biot** | $Bi$ | $Bi = \frac{h L_c}{k_{\text{solid}}}$ | $\frac{\text{Internal Thermal Conduction}}{\text{Surface Convection Resistance}}$ | Battery thermal management |
| **Damköhler** | $Da$ | $Da = \frac{\tau_{\text{flow}}}{\tau_{\text{reaction}}} = k_{\text{chem}} \tau_{\text{res}}$ | $\frac{\text{Chemical Reaction Rate}}{\text{Advective Transport Rate}}$ | Combustion engines, carbon plumes |

---

## 3. Tensor Analysis & Invariant Indices

In solid mechanics and continuum deformation, the following tensor invariants are critical:

- **Cauchy Stress Tensor:**
  $$\boldsymbol{\sigma} = \begin{bmatrix} \sigma_{xx} & \tau_{xy} & \tau_{xz} \\ \tau_{yx} & \sigma_{yy} & \tau_{yz} \\ \tau_{zx} & \tau_{zy} & \sigma_{zz} \end{bmatrix}$$
- **First Stress Invariant (Hydrostatic Trace):**
  $$I_1 = \text{Tr}(\boldsymbol{\sigma}) = \sigma_{xx} + \sigma_{yy} + \sigma_{zz} = 3 \sigma_m$$
- **Deviatoric Stress Tensor:**
  $$\mathbf{s} = \boldsymbol{\sigma} - \frac{1}{3} I_1 \mathbf{I}$$
- **Second Deviatoric Invariant ($J_2$) & von Mises Equivalent Stress:**
  $$J_2 = \frac{1}{2} \mathbf{s} : \mathbf{s} = \frac{1}{6}\left[ (\sigma_{xx} - \sigma_{yy})^2 + (\sigma_{yy} - \sigma_{zz})^2 + (\sigma_{zz} - \sigma_{xx})^2 + 6(\tau_{xy}^2 + \tau_{yz}^2 + \tau_{zx}^2) \right]$$
  $$\sigma_{\text{vonMises}} = \sqrt{3 J_2}$$

> [!TIP] Frame Indifference Rule
> Any scalar neural network prediction describing material yield, plastic deformation, or fluid viscosity must be formulated strictly as a function of tensor invariants ($I_1, J_2, J_3$) rather than raw coordinate tensor components $(\sigma_{ij})$ to preserve objective material frame indifference.
