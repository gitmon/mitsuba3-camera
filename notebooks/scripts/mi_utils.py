import sys
sys.path.insert(0, '/home/chuah/mitsuba3-camera/build/python')
import drjit as dr
import mitsuba as mi
import numpy as np

def create_mesh(V: np.ndarray, F: np.ndarray, name: str):
    Nv = V.shape[0]
    Nf = F.shape[0]

    # Convert vertices and faces
    V = mi.Vector3f(V)
    F = mi.Vector3u(F)

    # Instantiate the mesh object
    mesh = mi.Mesh(name, Nv, Nf, has_vertex_texcoords=False)

    # Set its buffers
    mesh_params = mi.traverse(mesh)
    mesh_params['vertex_positions'] = dr.ravel(V)
    mesh_params['faces'] = dr.ravel(F)
    mesh_params.update()

    return mesh
