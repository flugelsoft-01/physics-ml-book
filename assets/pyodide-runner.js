/**
 * In-Browser WebAssembly Python Runner (Pyodide 0.26)
 * Data Science & Machine Learning for Physics
 * Author: Kalyanjit Hatibaruah | Publisher: Flugelsoft Labs (2026)
 */

let pyodideInstance = null;
let isPyodideLoading = false;
let currentChapterSlug = "";

function appendConsoleOutput(text, isError = false) {
  const terminal = document.getElementById("pyodide-terminal-output");
  if (!terminal) return;
  const line = document.createElement("div");
  line.className = isError ? "terminal-line error" : "terminal-line";
  line.textContent = text;
  terminal.appendChild(line);
  terminal.scrollTop = terminal.scrollHeight;
}

window.renderCapturedPlot = function(base64Data) {
  const plotContainer = document.getElementById("pyodide-plot-output");
  if (!plotContainer) return;
  plotContainer.innerHTML = "";
  const img = document.createElement("img");
  img.src = "data:image/png;base64," + base64Data;
  img.alt = "Generated Simulation Diagnostic Plot";
  img.className = "pyodide-plot-img";
  plotContainer.appendChild(img);
  plotContainer.style.display = "block";
};

async function ensurePyodideLoaded(statusEl) {
  if (pyodideInstance) return pyodideInstance;
  if (isPyodideLoading) {
    while (isPyodideLoading) {
      await new Promise(r => setTimeout(r, 100));
    }
    return pyodideInstance;
  }

  isPyodideLoading = true;
  if (statusEl) statusEl.textContent = "Loading WebAssembly Python (Pyodide)...";

  if (typeof loadPyodide === "undefined") {
    await new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js";
      script.onload = resolve;
      script.onerror = () => reject(new Error("Failed to load Pyodide CDN"));
      document.head.appendChild(script);
    });
  }

  if (statusEl) statusEl.textContent = "Booting Python VM & installing NumPy, SciPy, Matplotlib...";
  pyodideInstance = await loadPyodide({
    stdout: (text) => appendConsoleOutput(text),
    stderr: (text) => appendConsoleOutput(text, true)
  });

  await pyodideInstance.loadPackage(["numpy", "scipy", "matplotlib"]);

  // Configure matplotlib non-interactive backend & auto-plot capture
  pyodideInstance.runPython(`
import sys
import io
import base64
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _pyodide_show():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')
    import js
    js.renderCapturedPlot(img_b64)

plt.show = _pyodide_show
`);

  // Attempt to load shared_utils/physics_helpers.py if available
  try {
    const basePath = document.body.dataset.basepath || "./";
    const res = await fetch(basePath + "shared_utils/physics_helpers.py");
    if (res.ok) {
      const helperCode = await res.text();
      pyodideInstance.FS.mkdirTree("shared_utils");
      pyodideInstance.FS.writeFile("shared_utils/physics_helpers.py", helperCode);
      pyodideInstance.FS.writeFile("shared_utils/__init__.py", "");
    }
  } catch (e) {
    console.warn("Could not pre-populate shared_utils in Pyodide:", e);
  }

  isPyodideLoading = false;
  if (statusEl) statusEl.textContent = "✓ Ready! Click 'Run Code' or press Shift+Enter";
  return pyodideInstance;
}

async function executePyodideCode() {
  const codeEditor = document.getElementById("pyodide-code-editor");
  const statusEl = document.getElementById("pyodide-status-badge");
  const terminal = document.getElementById("pyodide-terminal-output");
  const plotContainer = document.getElementById("pyodide-plot-output");
  const runBtn = document.getElementById("pyodide-run-btn");

  if (!codeEditor) return;
  const code = codeEditor.value;

  if (terminal) terminal.innerHTML = "";
  if (plotContainer) {
    plotContainer.innerHTML = "";
    plotContainer.style.display = "none";
  }

  if (runBtn) {
    runBtn.disabled = true;
    runBtn.textContent = "⏳ Running...";
  }

  try {
    const py = await ensurePyodideLoaded(statusEl);
    if (statusEl) statusEl.textContent = "⚡ Executing simulation...";

    // Append auto-render plt.show() if matplotlib was imported
    let executableCode = code;
    if (executableCode.includes("matplotlib") && !executableCode.includes("plt.show()")) {
      executableCode += "\nimport matplotlib.pyplot as plt\nplt.show()\n";
    }

    await py.runPythonAsync(executableCode);
    if (statusEl) statusEl.textContent = "✓ Execution completed successfully.";
  } catch (err) {
    appendConsoleOutput(String(err), true);
    if (statusEl) statusEl.textContent = "❌ Execution Error";
  } finally {
    if (runBtn) {
      runBtn.disabled = false;
      runBtn.textContent = "▶ Run Code";
    }
  }
}

function initPyodideWorkbench() {
  const toggleBtn = document.getElementById("pyodide-toggle-btn");
  const workbench = document.getElementById("pyodide-workbench-panel");
  const closeBtn = document.getElementById("pyodide-close-btn");
  const runBtn = document.getElementById("pyodide-run-btn");
  const resetBtn = document.getElementById("pyodide-reset-btn");
  const codeEditor = document.getElementById("pyodide-code-editor");
  const statusEl = document.getElementById("pyodide-status-badge");

  if (!workbench) return;

  // Extract first python code block on page as initial snippet
  const firstPythonBlock = document.querySelector(".chapter-body pre code.language-python, .chapter-body pre code");
  let defaultCode = "# Python code for this capstone\nimport numpy as np\nprint('Ready to run Physics-ML simulation!')\n";
  if (firstPythonBlock) {
    defaultCode = firstPythonBlock.innerText.trim();
  }

  if (codeEditor) {
    codeEditor.value = defaultCode;
    codeEditor.dataset.originalCode = defaultCode;
  }

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      workbench.classList.toggle("open");
      if (workbench.classList.contains("open")) {
        ensurePyodideLoaded(statusEl);
        workbench.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    });
  }

  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      workbench.classList.remove("open");
    });
  }

  if (runBtn) {
    runBtn.addEventListener("click", executePyodideCode);
  }

  if (resetBtn && codeEditor) {
    resetBtn.addEventListener("click", () => {
      codeEditor.value = codeEditor.dataset.originalCode || "";
      const terminal = document.getElementById("pyodide-terminal-output");
      const plot = document.getElementById("pyodide-plot-output");
      if (terminal) terminal.innerHTML = "";
      if (plot) {
        plot.innerHTML = "";
        plot.style.display = "none";
      }
      if (statusEl) statusEl.textContent = "Reset to original chapter code.";
    });
  }

  // Keyboard shortcut: Shift+Enter runs code in editor
  if (codeEditor) {
    codeEditor.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && e.shiftKey) {
        e.preventDefault();
        executePyodideCode();
      }
    });
  }
}

document.addEventListener("DOMContentLoaded", initPyodideWorkbench);
