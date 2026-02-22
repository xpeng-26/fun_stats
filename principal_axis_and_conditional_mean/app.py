import streamlit as st
import numpy as np
from scipy.stats import multivariate_normal
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Principal Axis vs Conditional Mean",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; }

    /* Hide the sidebar toggle button entirely */
    button[data-testid="collapsedControl"] { display: none; }

    .param-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #555;
        margin-bottom: 2px;
    }
    .math-panel {
        background: #f5f7fa;
        border-left: 4px solid #888;
        border-radius: 6px;
        padding: 16px 20px;
        margin-top: 8px;
        font-size: 0.93rem;
        line-height: 1.7;
    }
    .line-pca { color: #d62728; font-weight: 600; }
    .line-reg { color: #2ca02c; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Page title ────────────────────────────────────────────────────────────────
st.title("Principal Axis vs Conditional Mean")
st.caption("Visualizing the difference between PCA direction and the regression line on a bivariate Gaussian.")

# ── Main layout: left panel | chart ──────────────────────────────────────────
left_col, chart_col = st.columns([1, 5])

with left_col:
    # ── Parameters ───────────────────────────────────────────────────────────
    st.markdown("#### Parameters")
    rho = st.number_input(
        "Correlation  ρ",
        min_value=0.00,
        max_value=0.99,
        value=0.00,
        step=0.05,
        format="%.2f",
        help="Pearson correlation between X and Y",
    )
    sigma_x = st.number_input(
        "Std Dev  σₓ",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        format="%.1f",
        help="Standard deviation of X",
    )
    sigma_y = st.number_input(
        "Std Dev  σᵧ",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        format="%.1f",
        help="Standard deviation of Y",
    )

    # ── Covariance matrix & eigendecomposition ────────────────────────────────
    cov = np.array(
        [
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal_vec = eigenvectors[:, -1]
    if principal_vec[0] < 0:
        principal_vec = -principal_vec

    # Degenerate only when rho=0 AND sigma_x==sigma_y (all directions equally valid)
    _pca_degenerate = (rho == 0.0 and sigma_x == sigma_y)
    _pca_vertical   = (not _pca_degenerate) and abs(principal_vec[0]) < 1e-9
    slope_pca = (float("inf")      if _pca_vertical
                 else 1.0          if _pca_degenerate
                 else principal_vec[1] / principal_vec[0])
    slope_reg = rho * (sigma_y / sigma_x)

    # ── Slopes ────────────────────────────────────────────────────────────────
    st.markdown("#### Slopes")
    st.markdown(
        f"""
        <div class="math-panel">
          <span class="line-pca">■ Principal Axis (PCA)</span><br>
          slope = <b>{'—' if _pca_degenerate else ('∞' if _pca_vertical else f'{slope_pca:.4f}')}</b><br><br>
          <span class="line-reg">■ Regression Line E[Y|X]</span><br>
          slope = <b>{slope_reg:.4f}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Color controls ──────────────────────────────────────────────────────────
    st.markdown("#### Colors")

    _COLORSCALES = [
        "Blues", "Blues_r", "Viridis", "Greys_r", "Greys", "GnBu_r", "GnBu", "Plasma", "Inferno",
        "Magma", "Cividis", "Turbo", "Jet", "Hot", "Electric",
        "Portland", "Blackbody", "Earth", "Rainbow",
        "YlOrRd", "YlGnBu", "BuGn", "BuPu", "GnBu", "OrRd",
        "PuBu", "PuBuGn", "PuRd", "RdPu", "Greens", "Reds",
        "Purples", "Oranges", "RdBu", "RdYlBu", "Spectral",
    ]
    surface_colorscale = st.selectbox("Surface", options=_COLORSCALES, index=0)
    color_pca = st.color_picker("Principal Axis", "#d62728")
    color_reg = st.color_picker("Regression Line", "#2ca02c")

# ── Gaussian surface computation ──────────────────────────────────────────────
extent_x = 25
extent_y = 25
n_grid = 80

x_vals = np.linspace(-extent_x, extent_x, n_grid)
y_vals = np.linspace(-extent_y, extent_y, n_grid)
X, Y = np.meshgrid(x_vals, y_vals)
pos = np.dstack((X, Y))
rv = multivariate_normal(mean=[0, 0], cov=cov)
Z = rv.pdf(pos)

# ── Build figure — STABLE TRACE COUNT (always 3 traces) ──────────────────────
# Stable trace count is critical for uirevision to preserve camera across reruns.
# Variable trace count (e.g. from matplotlib contour segments) breaks uirevision.
fig = go.Figure()

# Trace 1 — Surface with built-in z-contours (no extra traces needed)
fig.add_trace(
    go.Surface(
        x=x_vals,
        y=y_vals,
        z=Z,
        colorscale=surface_colorscale,
        opacity=1.0,
        showscale=True,
        lighting=dict(ambient=0.9, diffuse=0.0, roughness=0.9, specular=0.01),
        contours_z=dict(
            show=True,
            usecolormap=False,
            highlight=True,
            highlightwidth=16,
            color="white",
            width=3,
            start=Z.max() * 0.05,
            end=Z.max() * 0.95,
            size=(Z.max() * 0.90) / 4,
        ),
        name="Gaussian PDF",
    )
)

# ── Line data ─────────────────────────────────────────────────────────────────
line_x = np.linspace(-extent_x, extent_x, 1000)

# PCA line
if not _pca_degenerate and not _pca_vertical:
    line_y_pca = slope_pca * line_x
elif _pca_vertical:
    # vertical: sweep y, fix x=0 — reuse line_x as y-axis parameter below
    line_y_pca = line_x.copy()
    line_x_pca_v = np.zeros_like(line_x)

if not _pca_degenerate:
    if _pca_vertical:
        _pts_pca = np.column_stack([line_x_pca_v, line_y_pca])
        _z_pca   = rv.pdf(_pts_pca) + 0.002 * Z.max()
        pca_x, pca_y, pca_z = line_x_pca_v, line_y_pca, _z_pca
    else:
        line_y_pca = slope_pca * line_x
        mask_pca   = (line_y_pca >= -extent_y) & (line_y_pca <= extent_y)
        _pts_pca   = np.column_stack([line_x[mask_pca], line_y_pca[mask_pca]])
        _z_pca     = rv.pdf(_pts_pca) + 0.002 * Z.max()
        pca_x, pca_y, pca_z = line_x[mask_pca], line_y_pca[mask_pca], _z_pca

# Regression line
line_y_reg = slope_reg * line_x
mask_reg   = (line_y_reg >= -extent_y) & (line_y_reg <= extent_y)
_pts_reg   = np.column_stack([line_x[mask_reg], line_y_reg[mask_reg]])
_z_reg     = rv.pdf(_pts_reg) + 0.002 * Z.max()

# Trace 2 — PCA axis (dummy trace when degenerate to keep trace count = 3)
if not _pca_degenerate:
    fig.add_trace(
        go.Scatter3d(
            x=pca_x, y=pca_y, z=pca_z,
            mode="lines",
            line=dict(color=color_pca, width=8),
            name="Principal Axis (PCA)",
        )
    )
else:
    fig.add_trace(
        go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="lines",
            line=dict(color=color_pca, width=8),
            name="Principal Axis (undefined)",
        )
    )

# Trace 3 — Regression line (always present)
fig.add_trace(
    go.Scatter3d(
        x=line_x[mask_reg],
        y=line_y_reg[mask_reg],
        z=_z_reg,
        mode="lines",
        line=dict(color=color_reg, width=8),
        name="Regression Line E[Y|X]",
    )
)

# ── Persistent view logic ─────────────────────────────────────────────────────
if "view_initialized" not in st.session_state:
    st.session_state.view_initialized = False

layout_dict = {
    "uirevision": "constant_key",   # never changes → Plotly JS preserves camera
    "template": "plotly_white",
    "margin": dict(l=0, r=0, t=0, b=0),
    "height": 800,
    "scene": {
        "xaxis_title": "X",
        "yaxis_title": "Y",
        "zaxis_title": "PDF",
        "aspectmode": "manual",
        "aspectratio": dict(x=1.4, y=1.4, z=1),
        "dragmode": "turntable",
    },
    "legend": dict(
        x=0.02, y=0.02,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#cccccc",
        borderwidth=1,
    ),
}

# Force initial top-down camera ONLY on first load
if not st.session_state.view_initialized:
    layout_dict["scene"]["camera"] = dict(
        eye=dict(x=0, y=0, z=2.5),
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
    )
    st.session_state.view_initialized = True

fig.update_layout(**layout_dict)

with chart_col:
    st.plotly_chart(fig, use_container_width=True, key="fixed_stats_plot", theme=None)
    if _pca_degenerate:
        st.caption("When ρ = 0 and σₓ = σᵧ, the distribution is isotropic and the"
                   " principal axis is undefined — the line is not shown.")

# ── Why Do They Differ? ───────────────────────────────────────────────────────
st.markdown("#### Why Do They Differ?")
