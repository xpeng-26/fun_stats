import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from scipy.stats import multivariate_normal

# ── App setup ─────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Principal Axis vs Conditional Mean")
server = app.server  # For deployment if needed

# ── Constants ────────────────────────────────────────────────────────────────
EXTENT = 25
N_GRID = 70
COLORSCALES = [
    "Blues", "Blues_r", "Viridis", "Greys_r", "Greys", "GnBu_r", "GnBu", "Plasma", "Inferno",
    "Magma", "Cividis", "Turbo", "Jet", "Hot", "Electric", "Portland", "Blackbody",
    "Earth", "Rainbow", "YlOrRd", "YlGnBu", "BuGn", "BuPu", "OrRd",
    "PuBu", "PuBuGn", "PuRd", "RdPu", "Greens", "Reds",
    "Purples", "Oranges", "RdBu", "RdYlBu", "Spectral",
]
INITIAL_CAMERA = dict(
    eye=dict(x=0, y=0, z=2.5),
    up=dict(x=1, y=0, z=0),
    center=dict(x=0, y=0, z=0),
)

# ── CSS ──────────────────────────────────────────────────────────────────────
PANEL_STYLE = {
    "background": "#f5f7fa",
    "borderLeft": "4px solid #888",
    "borderRadius": "6px",
    "padding": "14px 18px",
    "marginTop": "10px",
    "fontSize": "1.2rem",
    "lineHeight": "1.8",
}
LABEL_STYLE = {
    "fontSize": "1.2rem",
    "fontWeight": "600",
    "color": "#555",
    "marginBottom": "2px",
    "marginTop": "10px",
    "display": "block",
}
SECTION_STYLE = {"fontWeight": "700", "fontSize": "1.5rem", "marginTop": "50px", "marginBottom": "4px"}
COLOR_ROW = {"display": "flex", "alignItems": "center", "gap": "8px", "marginTop": "6px"}

# ── Layout ───────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"display": "flex", "flexDirection": "column", "fontFamily": "sans-serif",
           "background": "#ffffff", "minHeight": "100vh"},
    children=[
        # ── Top header ──────────────────────────────────────────────────────
        html.Div(
            style={"padding": "16px 24px 8px 24px", "borderBottom": "1px solid #e0e0e0"},
            children=[
                html.H2("Principal Axis vs Conditional Mean",
                        style={"fontSize": "2.0rem", "margin": "0", "lineHeight": "1.3"}),
                html.P("PCA direction vs. Regression line on a bivariate Gaussian.",
                       style={"fontSize": "1.2rem", "color": "#777", "margin": "4px 0 0 0"}),
            ],
        ),

        # ── Main body: left panel + chart ───────────────────────────────────
        html.Div(
            style={"display": "flex", "flexDirection": "row", "flex": "1"},
            children=[
                # ── Left panel ───────────────────────────────────────────────
                html.Div(
                    style={"width": "400px", "flexShrink": "0", "padding": "20px 16px",
                           "background": "#fafafa", "borderRight": "1px solid #e0e0e0",
                           "overflowY": "auto"},
                    children=[

                # Parameters
                html.Div("Parameters", style=SECTION_STYLE),

                html.Label("Correlation ρ", style=LABEL_STYLE),
                dcc.Slider(id="rho", min=0.0, max=0.99, step=0.05, value=0.0,
                           marks={
                                0: {"label": "0", "style": {"fontSize": "1rem"}},
                                0.5: {"label": "0.5", "style": {"fontSize": "1rem"}},
                                0.99: {"label": "0.99", "style": {"fontSize": "1rem"}}
                            },
                           tooltip={"placement": "bottom", "always_visible": True}),

                html.Label("Std Dev σₓ", style=LABEL_STYLE),
                dcc.Slider(id="sigma-x", min=1.0, max=10.0, step=0.5, value=5.0,
                           marks={
                                1: {"label": "1", "style": {"fontSize": "1rem"}},
                                5: {"label": "5", "style": {"fontSize": "1rem"}},
                                10: {"label": "10", "style": {"fontSize": "1rem"}}
                            },
                            tooltip={"placement": "bottom", "always_visible": True}),

                html.Label("Std Dev σᵧ", style=LABEL_STYLE),
                dcc.Slider(id="sigma-y", min=1.0, max=10.0, step=0.5, value=5.0,
                           marks={
                                1: {"label": "1", "style": {"fontSize": "1rem"}},
                                5: {"label": "5", "style": {"fontSize": "1rem"}},
                                10: {"label": "10", "style": {"fontSize": "1rem"}}
                            },
                           tooltip={"placement": "bottom", "always_visible": True}),

                # Slopes display
                html.Div("Slopes", style=SECTION_STYLE),
                html.Div(id="slopes-panel", style=PANEL_STYLE),

                # Colors
                html.Div("Colors", style=SECTION_STYLE),

                html.Label("Surface", style=LABEL_STYLE),
                dcc.Dropdown(id="colorscale", options=[{"label": c, "value": c} for c in COLORSCALES],
                             value="Blues", clearable=False,
                             style={"fontSize": "1.2rem"}),

                html.Label("Principal Axis", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="color-pca",
                    options=[
                        {"label": "🔴 Red",    "value": "#d62728"},
                        {"label": "🟠 Orange", "value": "#ff7f0e"},
                        {"label": "🟡 Yellow", "value": "#f0c040"},
                        {"label": "🟢 Green",  "value": "#2ca02c"},
                        {"label": "🔵 Blue",   "value": "#1f77b4"},
                        {"label": "🟣 Purple", "value": "#9467bd"},
                        {"label": "🩷 Pink",   "value": "#e377c2"},
                        {"label": "⚫ Black",  "value": "#222222"},
                    ],
                    value="#d62728", clearable=False,
                    style={"fontSize": "1.2rem"},
                ),

                html.Label("Regression Line", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="color-reg",
                    options=[
                        {"label": "🔴 Red",    "value": "#d62728"},
                        {"label": "🟠 Orange", "value": "#ff7f0e"},
                        {"label": "🟡 Yellow", "value": "#f0c040"},
                        {"label": "🟢 Green",  "value": "#2ca02c"},
                        {"label": "🔵 Blue",   "value": "#1f77b4"},
                        {"label": "🟣 Purple", "value": "#9467bd"},
                        {"label": "🩷 Pink",   "value": "#e377c2"},
                        {"label": "⚫ Black",  "value": "#222222"},
                    ],
                    value="#2ca02c", clearable=False,
                    style={"fontSize": "1.2rem"},
                ),
            ],
        ),

        # ── Chart area ──────────────────────────────────────────────────────
        html.Div(
            style={"flex": "1", "display": "flex", "flexDirection": "column"},
            children=[
                dcc.Graph(
                    id="graph",
                    style={"height": "1000px"},
                    config={"scrollZoom": True},
                ),
                html.Div(id="footnote",
                         style={"padding": "4px 16px", "fontSize": "1.2rem", "color": "#888"}),
            ],
            ),  # end chart area
        ],      # end main body children
        ),      # end main body div

        # Camera store — persists user's dragged view
        dcc.Store(id="camera-store", data=None),
    ],
)


# ── Callback 1: capture camera when user rotates ──────────────────────────────
@app.callback(
    Output("camera-store", "data"),
    Input("graph", "relayoutData"),
    State("camera-store", "data"),
    prevent_initial_call=True,
)
def save_camera(relayout_data, current_camera):
    if relayout_data and "scene.camera" in relayout_data:
        return relayout_data["scene.camera"]
    return current_camera


# ── Callback 2: update figure & slopes ────────────────────────────────────────
@app.callback(
    Output("graph", "figure"),
    Output("slopes-panel", "children"),
    Output("footnote", "children"),
    Input("rho", "value"),
    Input("sigma-x", "value"),
    Input("sigma-y", "value"),
    Input("colorscale", "value"),
    Input("color-pca", "value"),
    Input("color-reg", "value"),
    State("camera-store", "data"),
)
def update_figure(rho, sigma_x, sigma_y, colorscale, color_pca, color_reg, camera):
    rho     = rho     or 0.0
    sigma_x = sigma_x or 5.0
    sigma_y = sigma_y or 5.0

    # ── Covariance & eigenvectors ─────────────────────────────────────────
    cov = np.array([
        [sigma_x**2,              rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2             ],
    ])
    _, eigenvectors = np.linalg.eigh(cov)
    principal_vec = eigenvectors[:, -1]
    if principal_vec[0] < 0:
        principal_vec = -principal_vec

    _pca_degenerate = (rho == 0.0 and sigma_x == sigma_y)
    _pca_vertical   = (not _pca_degenerate) and abs(principal_vec[0]) < 1e-9
    slope_pca = (float("inf")           if _pca_vertical
                 else 1.0               if _pca_degenerate
                 else principal_vec[1] / principal_vec[0])
    slope_reg = rho * (sigma_y / sigma_x)

    # ── Gaussian grid ─────────────────────────────────────────────────────
    x_vals = np.linspace(-EXTENT, EXTENT, N_GRID)
    y_vals = np.linspace(-EXTENT, EXTENT, N_GRID)
    X, Y   = np.meshgrid(x_vals, y_vals)
    pos    = np.dstack((X, Y))
    rv     = multivariate_normal(mean=[0, 0], cov=cov)
    Z      = rv.pdf(pos)

    # ── Trace 1: Surface (always present) ─────────────────────────────────
    surface_trace = go.Surface(
        x=x_vals, y=y_vals, z=Z,
        colorscale=colorscale,
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

    # ── Trace 2: PCA axis or dummy (stable trace count) ───────────────────
    line_x = np.linspace(-EXTENT, EXTENT, 1000)
    if not _pca_degenerate:
        if _pca_vertical:
            pca_lx = np.zeros_like(line_x)
            pca_ly = line_x.copy()
        else:
            pca_ly = slope_pca * line_x
            mask   = (pca_ly >= -EXTENT) & (pca_ly <= EXTENT)
            line_x = line_x[mask]
            pca_ly = pca_ly[mask]
            pca_lx = line_x
        pts_pca = np.column_stack([pca_lx, pca_ly])
        pca_z   = rv.pdf(pts_pca) + 0.002 * Z.max()
        pca_trace = go.Scatter3d(
            x=pca_lx, y=pca_ly, z=pca_z,
            mode="lines", line=dict(color=color_pca, width=8),
            name="Principal Axis (PCA)",
        )
    else:
        pca_trace = go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="lines", line=dict(color=color_pca, width=8),
            name="Principal Axis (undefined)",
        )

    # ── Trace 3: Regression line (always present) ─────────────────────────
    line_x_full = np.linspace(-EXTENT, EXTENT, 1000)
    reg_ly  = slope_reg * line_x_full
    mask_r  = (reg_ly >= -EXTENT) & (reg_ly <= EXTENT)
    pts_reg = np.column_stack([line_x_full[mask_r], reg_ly[mask_r]])
    reg_z   = rv.pdf(pts_reg) + 0.002 * Z.max()
    reg_trace = go.Scatter3d(
        x=line_x_full[mask_r], y=reg_ly[mask_r], z=reg_z,
        mode="lines", line=dict(color=color_reg, width=8),
        name="Regression Line E[Y|X]",
    )

    # ── Figure ────────────────────────────────────────────────────────────
    fig = go.Figure(data=[surface_trace, pca_trace, reg_trace])

    # Restore user's camera; fall back to initial top-down view
    scene_camera = camera if camera else INITIAL_CAMERA

    fig.update_layout(
        uirevision="constant_key",
        template="plotly_white",
        margin=dict(l=0, r=0, t=0, b=0),
        height=1000,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="PDF",
            xaxis={"title": {"font": {"size": 16}}, "tickfont": {"size": 13}},
            yaxis={"title": {"font": {"size": 16}}, "tickfont": {"size": 13}},
            zaxis={"title": {"font": {"size": 16}}, "tickfont": {"size": 13}},
            aspectmode="manual",
            aspectratio=dict(x=1.4, y=1.4, z=1),
            dragmode="turntable",
            camera=scene_camera,
        ),
        legend=dict(
            x=0.01, y=0.01,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
    )

    # ── Slopes panel ──────────────────────────────────────────────────────
    slope_pca_str = ("—" if _pca_degenerate
                     else "∞" if _pca_vertical
                     else f"{slope_pca:.2f}")
    slopes_content = [
        html.Span("■ Principal Axis (PCA)",
                  style={"color": color_pca, "fontWeight": "600"}),
        html.Br(),
        html.Span(f"slope = "), html.B(slope_pca_str),
        html.Br(), html.Br(),
        html.Span("■ Regression Line E[Y|X]",
                  style={"color": color_reg, "fontWeight": "600"}),
        html.Br(),
        html.Span("slope = "), html.B(f"{slope_reg:.2f}"),
    ]

    # ── Footnote ──────────────────────────────────────────────────────────
    footnote = (
        "When ρ = 0 and σₓ = σᵧ, the distribution is isotropic and the "
        "principal axis is undefined — the line is not shown."
        if _pca_degenerate else ""
    )

    return fig, slopes_content, footnote


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)
