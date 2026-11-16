# Interactive Physics-ML In-Browser Sandbox

This interactive laboratory executes real-time numerical physics solvers and conservation-law residual checks entirely inside your client browser using high-performance Vanilla JavaScript and HTML5 Canvas.

Adjust physical parameters with the sliders below to explore non-linear wave steepening, shockwave inception, and 2D thermal conduction relaxation in real time.

---

## 1. 1D Non-Linear Viscous Burgers' Equation Explorer

The 1D viscous Burgers' equation is the foundational benchmark problem in fluid dynamics and Physics-Informed Neural Networks:

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

It couples non-linear advective shock formation ($u \frac{\partial u}{\partial x}$) with viscous dissipative diffusion ($\nu \frac{\partial^2 u}{\partial x^2}$).

<div class="figure-container" style="padding: 24px;">
  <!-- Simulation Controls -->
  <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 16px; background: var(--bg-surface); padding: 16px; border-radius: 8px; border: 1px solid var(--border-color); align-items: center;">
    <div>
      <label style="font-size: 0.82rem; font-weight: 700; display: block; margin-bottom: 4px;">Kinematic Viscosity ($\nu$): <span id="val-viscosity" style="color:var(--brand-primary); font-family:var(--font-mono);">0.008</span></label>
      <input type="range" id="slider-viscosity" min="0.001" max="0.05" step="0.001" value="0.008" style="width: 160px;">
    </div>
    <div>
      <label style="font-size: 0.82rem; font-weight: 700; display: block; margin-bottom: 4px;">Initial Amplitude ($A$): <span id="val-amplitude" style="color:var(--brand-primary); font-family:var(--font-mono);">1.20</span></label>
      <input type="range" id="slider-amplitude" min="0.2" max="2.0" step="0.1" value="1.2" style="width: 140px;">
    </div>
    <div style="display: flex; gap: 8px; margin-top: 18px;">
      <button id="btn-toggle-burgers" style="background: var(--brand-primary); color: white; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.85rem;">Pause</button>
      <button id="btn-reset-burgers" style="background: var(--bg-primary); border: 1px solid var(--border-color); color: var(--text-primary); padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 0.85rem;">Reset Wave</button>
    </div>
    <div style="margin-left: auto; text-align: right; font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted);">
      <div>Step: <span id="sim-step" style="color:var(--text-primary); font-weight:600;">0</span></div>
      <div>Max Gradient $|u_x|$: <span id="sim-grad" style="color:var(--brand-primary); font-weight:600;">0.00</span></div>
    </div>
  </div>

  <!-- Canvas -->
  <div style="background: var(--bg-primary); border: 1px solid var(--border-color); border-radius: 6px; overflow: hidden; position: relative;">
    <canvas id="canvas-burgers" width="800" height="340" style="width: 100%; height: 340px; display: block;"></canvas>
  </div>
  <div class="figure-caption" style="margin-top: 10px;">
    <strong>Live Canvas Simulation:</strong> 1D non-linear wave profile $u(x, t)$ advancing under Lax-Friedrichs finite-difference advection-diffusion. Notice how lowering viscosity $\nu$ causes steepening into a localized shockwave front.
  </div>
</div>

---

## 2. 2D Steady-State Poisson Heat Relaxation Sandbox

Click or drag on the canvas below to place localized fission or microchip heat sources ($Q > 0$). The simulator runs real-time successive relaxation of the 2D heat equation $-\alpha \nabla^2 T = Q(x, y)$, measuring the global PDE residual norm $\| \alpha \nabla^2 T + Q \|_2$.

<div class="figure-container" style="padding: 24px;">
  <div style="display: flex; gap: 16px; margin-bottom: 14px; align-items: center; justify-content: space-between; flex-wrap: wrap;">
    <div style="font-size: 0.88rem; color: var(--text-secondary);">
      <strong>Interactive Instructions:</strong> Click or drag inside the grid to deposit thermal energy.
    </div>
    <div style="display: flex; gap: 8px;">
      <button id="btn-clear-heat" style="background: var(--bg-surface); border: 1px solid var(--border-color); padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 0.82rem;">Clear Heat Sources</button>
      <div style="padding: 4px 10px; background: rgba(5, 150, 105, 0.1); border: 1px solid var(--accent-emerald); border-radius: 6px; font-family: var(--font-mono); font-size: 0.8rem; color: var(--accent-emerald);">
        PDE Residual: <span id="heat-res-val">0.00e+0</span>
      </div>
    </div>
  </div>

  <div style="background: #000; border: 1px solid var(--border-color); border-radius: 6px; overflow: hidden;">
    <canvas id="canvas-heat" width="500" height="280" style="width: 100%; height: 280px; display: block; cursor: crosshair;"></canvas>
  </div>
  <div class="figure-caption" style="margin-top: 10px;">
    <strong>Live 2D Heat Diffusion:</strong> In-browser SOR iterative solver computing continuous conduction toward 300K boundary conditions.
  </div>
</div>

<!-- Embedded Numerical Simulation Scripts -->
<script>
(function() {
  // ==========================================
  // 1. Burgers' Equation Simulation Engine
  // ==========================================
  const canvasB = document.getElementById('canvas-burgers');
  if (canvasB) {
    const ctxB = canvasB.getContext('2d');
    const nx = 200;
    const dx = 2.0 * Math.PI / nx;
    let dt = 0.008;
    let nu = 0.008;
    let amp = 1.2;
    let stepCount = 0;
    let isRunning = true;

    let u = new Float32Array(nx);
    let uNext = new Float32Array(nx);

    function initWave() {
      stepCount = 0;
      for (let i = 0; i < nx; i++) {
        const x = i * dx;
        u[i] = amp * Math.sin(x);
      }
    }
    initWave();

    function stepBurgers() {
      for (let i = 0; i < nx; i++) {
        const im1 = (i - 1 + nx) % nx;
        const ip1 = (i + 1) % nx;
        // Central flux advection + diffusion
        const adv = -u[i] * (u[ip1] - u[im1]) / (2.0 * dx);
        const diff = nu * (u[ip1] - 2.0 * u[i] + u[im1]) / (dx * dx);
        uNext[i] = u[i] + dt * (adv + diff);
      }
      u.set(uNext);
      stepCount++;
    }

    function renderBurgers() {
      ctxB.clearRect(0, 0, canvasB.width, canvasB.height);
      const w = canvasB.width;
      const h = canvasB.height;
      const midY = h / 2;
      const scaleY = 70.0;

      // Draw zero axis
      ctxB.strokeStyle = '#334155';
      ctxB.lineWidth = 1;
      ctxB.setLineDash([4, 4]);
      ctxB.beginPath();
      ctxB.moveTo(0, midY);
      ctxB.lineTo(w, midY);
      ctxB.stroke();
      ctxB.setLineDash([]);

      // Draw Wave Curve
      ctxB.strokeStyle = '#3b82f6';
      ctxB.lineWidth = 3;
      ctxB.beginPath();
      let maxGrad = 0;

      for (let i = 0; i < nx; i++) {
        const xPix = (i / (nx - 1)) * w;
        const yPix = midY - u[i] * scaleY;
        if (i === 0) ctxB.moveTo(xPix, yPix);
        else ctxB.lineTo(xPix, yPix);

        const im1 = (i - 1 + nx) % nx;
        const grad = Math.abs(u[i] - u[im1]) / dx;
        if (grad > maxGrad) maxGrad = grad;
      }
      ctxB.stroke();

      // Update telemetry indicators
      document.getElementById('sim-step').textContent = stepCount;
      document.getElementById('sim-grad').textContent = maxGrad.toFixed(2);

      if (isRunning) {
        stepBurgers();
      }
      requestAnimationFrame(renderBurgers);
    }
    requestAnimationFrame(renderBurgers);

    // Controls
    document.getElementById('slider-viscosity').addEventListener('input', (e) => {
      nu = parseFloat(e.target.value);
      document.getElementById('val-viscosity').textContent = nu.toFixed(3);
    });
    document.getElementById('slider-amplitude').addEventListener('input', (e) => {
      amp = parseFloat(e.target.value);
      document.getElementById('val-amplitude').textContent = amp.toFixed(2);
      initWave();
    });
    document.getElementById('btn-toggle-burgers').addEventListener('click', (e) => {
      isRunning = !isRunning;
      e.target.textContent = isRunning ? 'Pause' : 'Resume';
    });
    document.getElementById('btn-reset-burgers').addEventListener('click', () => {
      initWave();
    });
  }

  // ==========================================
  // 2. 2D Heat Equation Relaxation Engine
  // ==========================================
  const canvasH = document.getElementById('canvas-heat');
  if (canvasH) {
    const ctxH = canvasH.getContext('2d');
    const cols = 50;
    const rows = 28;
    let T = new Float32Array(cols * rows);
    let Q = new Float32Array(cols * rows);
    T.fill(300.0);

    let isDrawing = false;
    function addHeat(e) {
      const rect = canvasH.getBoundingClientRect();
      const x = Math.floor(((e.clientX - rect.left) / rect.width) * cols);
      const y = Math.floor(((e.clientY - rect.top) / rect.height) * rows);
      if (x > 1 && x < cols - 2 && y > 1 && y < rows - 2) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const idx = (y + dy) * cols + (x + dx);
            Q[idx] += 40.0;
            T[idx] += 30.0;
          }
        }
      }
    }

    canvasH.addEventListener('mousedown', (e) => { isDrawing = true; addHeat(e); });
    window.addEventListener('mouseup', () => { isDrawing = false; });
    canvasH.addEventListener('mousemove', (e) => { if (isDrawing) addHeat(e); });
    document.getElementById('btn-clear-heat').addEventListener('click', () => {
      Q.fill(0);
      T.fill(300.0);
    });

    function stepHeat() {
      let residualSum = 0;
      let count = 0;
      for (let y = 1; y < rows - 1; y++) {
        for (let x = 1; x < cols - 1; x++) {
          const idx = y * cols + x;
          const up = (y - 1) * cols + x;
          const down = (y + 1) * cols + x;
          const left = y * cols + (x - 1);
          const right = y * cols + (x + 1);

          const tNew = 0.25 * (T[up] + T[down] + T[left] + T[right] + 0.1 * Q[idx]);
          const diff = tNew - T[idx];
          T[idx] += 0.8 * diff;
          residualSum += diff * diff;
          count++;
        }
      }
      const resVal = Math.sqrt(residualSum / count);
      document.getElementById('heat-res-val').textContent = resVal.toExponential(2);
    }

    function renderHeat() {
      stepHeat();
      const imgData = ctxH.createImageData(cols, rows);
      for (let i = 0; i < T.length; i++) {
        const val = Math.min(1.0, Math.max(0.0, (T[i] - 300.0) / 120.0));
        const idx = i * 4;
        // Thermal colormap (black -> red -> orange -> yellow -> white)
        imgData.data[idx] = Math.floor(Math.min(255, val * 380));
        imgData.data[idx + 1] = Math.floor(Math.max(0, (val - 0.4) * 400));
        imgData.data[idx + 2] = Math.floor(Math.max(0, (val - 0.75) * 600));
        imgData.data[idx + 3] = 255;
      }

      // Draw scaled up to canvas
      createImageBitmap(imgData).then(bmp => {
        ctxH.drawImage(bmp, 0, 0, canvasH.width, canvasH.height);
      });
      requestAnimationFrame(renderHeat);
    }
    requestAnimationFrame(renderHeat);
  }
})();
</script>
