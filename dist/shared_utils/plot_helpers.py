"""
Plotting and Visualization Helpers: Matplotlib publication styling and Plotly interactive snippets.
"""
import os
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio


def set_academic_matplotlib_style():
    """Applies clean, high-DPI academic styling to Matplotlib figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'DejaVu Sans', 'Arial'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def save_matplotlib_figure(fig: plt.Figure, filepath: str) -> str:
    """Saves a Matplotlib figure ensuring parent directory exists and closes figure."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return filepath


def export_plotly_html_snippet(
    fig: go.Figure,
    div_id: Optional[str] = None,
    include_plotlyjs: bool = False
) -> str:
    """
    Exports a Plotly figure as a clean, responsive HTML div snippet.
    `include_plotlyjs=False` ensures fast page loads because the main layout imports Plotly once via CDN.
    """
    # Modern clean styling
    fig.update_layout(
        template='plotly_white',
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family='system-ui, -apple-system, sans-serif', size=12),
        hoverlabel=dict(font_size=12, font_family='system-ui, -apple-system, sans-serif')
    )
    
    config = {
        'responsive': True,
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    }
    
    html_snippet = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config=config,
        div_id=div_id
    )
    return html_snippet
