"""requirements: nilearn, plotly, kaleido"""
import plotly.graph_objects as go

from nilearn import datasets, surface
from nilearn.plotting.cm import cold_hot
from nilearn.plotting.js_plotting_utils import colorscale
from nilearn.plotting.html_surface import _get_vertexcolor

AXIS_CONFIG = {
    "showgrid": False,
    "showline": False,
    "ticks": "",
    "title": "",
    "showticklabels": False,
    "zeroline": False,
    "showspikes": False,
    "spikesides": False,
    "showbackground": False,
}

CAMERAS = {
    "left": {
        "eye": {"x": -1.7, "y": 0, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "right": {
        "eye": {"x": 1.7, "y": 0, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "top": {
        "eye": {"x": 0, "y": 0, "z": 1.7},
        "up": {"x": 0, "y": 1, "z": 0},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "bottom": {
        "eye": {"x": 0, "y": 0, "z": -1.7},
        "up": {"x": 0, "y": 1, "z": 0},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "front": {
        "eye": {"x": 0, "y": 1.7, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
    },
    "back": {
        "eye": {"x": 0, "y": -1.7, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
        "center": {"x": 0, "y": 0, "z": 0},
    },
}
LAYOUT = {
    "scene": {f"{dim}axis": AXIS_CONFIG for dim in ("x", "y", "z")},
    "paper_bgcolor": "#fff",
    "hovermode": False,
    "margin": {"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
}


def plot_surf(mesh, data, view="right", threshold="85%", output_file=None):
    coords, triangles = surface.load_surf_mesh(mesh)
    x, y, z = coords.T
    i, j, k = triangles.T
    colors = colorscale(cold_hot, data, threshold)
    vertexcolor = _get_vertexcolor(
        data,
        colors["cmap"],
        colors["norm"],
        colors["abs_threshold"],
        surface.load_surf_data(fsaverage["sulc_right"]),
    )
    mesh_3d = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, vertexcolor=vertexcolor)
    fig = go.Figure(data=[mesh_3d])
    fig.update_layout(scene_camera=CAMERAS[view], **LAYOUT)
    if output_file is not None:
        fig.write_image(output_file)
    return fig


if __name__ == "__main__":
    img = datasets.fetch_neurovault_motor_task()["images"][0]
    fsaverage = datasets.fetch_surf_fsaverage()
    mesh = fsaverage["pial_right"]
    data = surface.vol_to_surf(img, fsaverage["pial_right"])
    fig = plot_surf(mesh, data, output_file="/tmp/surface_plot.png")
    fig.show()
