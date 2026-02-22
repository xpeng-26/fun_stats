import os
import streamlit.components.v1 as components

_component_func = components.declare_component(
    "plotly_camera_chart",
    path=os.path.dirname(os.path.abspath(__file__)),
)


def plotly_camera_chart(figure_json: str, camera: dict | None = None, key: str = "plotly_chart"):
    """
    Render a Plotly figure inside a custom component that:
    - Applies `camera` to the scene on every render (preserving view)
    - Returns the updated camera dict whenever the user rotates/pans

    Parameters
    ----------
    figure_json : str   JSON string from fig.to_json()
    camera      : dict  Previously saved scene.camera dict (or None for default)
    key         : str   Unique Streamlit component key

    Returns
    -------
    dict | None   New camera dict if the user moved the view, else None
    """
    return _component_func(
        figure_json=figure_json,
        camera=camera,
        key=key,
        default=None,
    )
