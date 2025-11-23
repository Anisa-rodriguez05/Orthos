from pathlib import Path
import argparse

import trimesh
import plotly.graph_objects as go


def load_mesh(stl_path: Path) -> trimesh.Trimesh:
    """
    Load an STL mesh from the given path using trimesh.
    """
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    if not stl_path.suffix.lower() == ".stl":
        raise ValueError(f"File must be .stl, got: {stl_path.suffix}")

    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        # trimesh can sometimes return a Scene; force to a single mesh
        mesh = mesh.dump().sum()
    return mesh


def plot_mesh(mesh: trimesh.Trimesh) -> None:
    """
    Plot the given mesh with Plotly as an interactive 3D figure.
    """
    verts = mesh.vertices
    faces = mesh.faces

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=1.0,
                flatshading=True,
            )
        ]
    )

    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Simple STL viewer using trimesh + Plotly (local version of your Colab script)."
    )
    parser.add_argument(
        "stl_path",
        type=str,
        help="Path to the STL file to visualize.",
    )
    args = parser.parse_args()

    stl_path = Path(args.stl_path)

    mesh = load_mesh(stl_path)
    plot_mesh(mesh)


if __name__ == "__main__":
    main()
