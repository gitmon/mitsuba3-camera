import numpy as np
import matplotlib.pyplot as plt
import meshplot as mp
from matplotlib.colors import hsv_to_rgb

def plot_cross_section_2d(points, edges, normals = None, **kwargs):
    """
    Plot the lens cross-section generated by RealisticLens::draw_cross_section()
    """
    p1 = points[edges[:,0]]
    p2 = points[edges[:,1]]
    plt.plot([p1[:,2], p2[:,2]], [p1[:,0], p2[:,0]], **kwargs)
    plt.xlabel("Z (mm)")
    plt.ylabel("X (mm)")
    plt.axis('equal')
    xlims = plt.xlim()
    plt.plot(xlims, [0.,0.], 'k--', label="axis")

    if normals is not None:
        plt.quiver(
            points[:,2], points[:,0], 
            normals[:,2], normals[:,0], 
            scale=20.0, color='r') #, width=3e-3, scale_units='width', units='width')

from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def lathe(points, n_theta=32):
    """
    Sweep a 2D sequence of points into a radially-symmetric 3D mesh, with the Z-axis
    chosen as the axis of symmetry. The input points should canonically lie in the
    x-z plane.
    """
    points_mesh = []
    faces_mesh = []
    num_points_2d = points.shape[0]
    points_mesh.append(points)
    for rot_id, theta in enumerate(np.linspace(0, 2.0 * np.pi, n_theta + 1)[1:]):
        R = Rotation.from_rotvec(np.array([0,0,theta]))
        points_rot = R.apply(points)
        points_mesh.append(points_rot)

        # build faces: one quad per pair of parallel edges
        for idx_2d in range(num_points_2d - 1):
            # quad indices
            v0 = rot_id * num_points_2d + idx_2d
            v1 = rot_id * num_points_2d + idx_2d + 1
            v2 = ((rot_id + 1) % n_theta) * num_points_2d + idx_2d
            v3 = ((rot_id + 1) % n_theta) * num_points_2d + idx_2d + 1
            faces_mesh += [[v0, v1, v2], [v1, v3, v2]]

    V = np.vstack(points_mesh)
    F = np.array(faces_mesh)
    return V, F


def lathe_scene(points, edges):
    """
    Build 3D meshes for all lenses in the scene. The `points` and `edges` are generated 
    by RealisticLens::draw_cross_section().
    """
    meshes = []
    Nv = points.shape[0]
    Ne = edges.shape[0]
    A = coo_matrix((np.ones(Ne), (edges[:,0], edges[:,1])), shape=(Nv,Nv))
    num_segments, labels = connected_components(A, return_labels=True)
    for segment_id in range(num_segments):
        points_seg = points[labels==segment_id]
        V, F = lathe(points_seg)
        meshes.append((V,F))
    return meshes


def plot_cross_section_3d(points, edges, camera_origin=[0,0,0], mp_plot=None):
    """
    Build and plot 3D meshes for all lenses in the scene.
    """
    # z_mean = np.mean(points[:,2])
    meshes = lathe_scene(points, edges)
    origin = np.array(camera_origin).reshape(1,3)

    if mp_plot is None:
        mp_plot = mp.plot(np.zeros(3))
    for mesh_id, mesh in enumerate(meshes):
        V, F = mesh
        c = np.ones_like(F).astype('float')
        c[:,0] = (mesh_id + 1) / (len(meshes) + 1)
        c[:,1] = 0.5
        c = hsv_to_rgb(c)
        shading = {
            "flat": True, 
            "wireframe": True, 
            "wire_width": 0.03, 
            "wire_color": "gray", 
            "line_width": 0.1,
            }
        mp_plot.add_mesh(V - origin, F, c=c, shading=shading)
    
    return mp_plot