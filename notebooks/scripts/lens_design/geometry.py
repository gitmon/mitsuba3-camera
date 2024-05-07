'''
Helper functions for initializing lens-related geometry.
'''
import numpy as np
import meshplot as mp
import matplotlib.pyplot as plt
from gpytoolbox import icosphere, remove_unreferenced, boundary_loops
from scipy.sparse import coo_matrix
import typing as tp

def face_vertex_adjacency_matrix(V: np.ndarray, F: np.ndarray):
    '''
    Construct the face<->vertex adjacency matrix for a triangle mesh. The rows are faces (Nr == Nf) and 
    columns are vertex indices (Nc == Nv). Right-multiplying by a size-Nv vector will distribute the 
    vector's per-vertex values onto the corresponding faces.
    Input: 
    - V: np.ndarray [Nv x 3]. Vertex position data.
    - F: np.ndarray [Nf x 3]. Triangle data.
    Output:
    - M_VF: sparse matrix, [Nf x Nv].
    '''
    assert V.shape[1] == 3
    assert F.shape[1] == 3
    assert (F.dtype == np.int32) or (F.dtype == np.uint32) or (F.dtype == np.int64) or (F.dtype == np.uint64)

    Nv = V.shape[0]
    Nf = F.shape[0]
    ff = np.repeat(np.arange(Nf), 3)
    vv = F.ravel()
    values = np.ones_like(vv)
    M_FV = coo_matrix((values, (ff,vv)), shape=(Nf,Nv))
    return M_FV

def get_hemisphere(N: int):
    '''
    Construct the triangle mesh of a +z-oriented hemisphere. The mesh is open at the boundary z = 0.
    Input: 
    - N: int. Mesh refinement level.
    Output:
    - V: np.ndarray [Nv x 3]. Mesh vertex position data.
    - F: np.ndarray [Nf x 3]. Mesh triangle data.
    '''
    V, F = icosphere(N)

    # next, proceed to delete vertices in the -z hemisphere (z < 0):
    avg_edge_length = np.sqrt((4.0 * np.pi) / F.shape[0])
    # select vertices behind z = 0
    V_mask = V[:,2] < -avg_edge_length / 2.0
    # build F-V adjacency matrix to map vertex data -> faces
    M_FV = face_vertex_adjacency_matrix(V, F)
    # select faces that contain *zero* masked (== invalid) vertices
    valid_faces_mask = (M_FV @ V_mask) == 0

    F_ = F[valid_faces_mask]
    # remove deleted/unreferenced vertices
    V_, F_ = remove_unreferenced(V, F_)

    return V_, F_

def get_spherical_cap(N, elem_radius: float, abs_curvature: float):
    '''
    Construct the triangle mesh of a +z-oriented spherical cap. The mesh is open at the boundary,
    {z = 0, r = elem_radius}. We compute the cap geometry as an angular rescaling of the unit 
    hemisphere.

    Input: 
    - N: int. Mesh refinement level.
    - elem_radius: float. Radial extent of the surface.
    - abs_curvature: float. The surface's 2nd-order curvature (or an estimate of it).
    Output:
    - V: np.ndarray [Nv x 3]. Mesh vertex position data.
    - F: np.ndarray [Nf x 3]. Mesh triangle data.

    '''
    V, F = get_hemisphere(N)

    target_angle = np.arcsin(elem_radius * max(abs_curvature, 1e-3))
    angle_rescale = target_angle / (0.5 * np.pi)

    V_ = V.copy()
    vlens = np.linalg.norm(V_, axis=1)#[:,None]
    angles = np.arccos(V_[:,2] / vlens.ravel()) # np.sum(V[:,:2] ** 2, axis=1)
    angles *= angle_rescale

    V_[:,2] = vlens * np.cos(angles)
    V_[:,:2] *= (vlens * np.sin(angles) / np.linalg.norm(V[:,:2], axis=1))[:, None]

    print("V_ lies on sphere:", np.allclose(np.linalg.norm(V_, axis=1), np.ones_like(vlens)))

    # shift and scale to correct size
    V_[:,2] -= np.cos(target_angle)
    V_ *= elem_radius / np.sin(target_angle)

    return V_, F


def create_lens_geometry(N: int, r_film: float, r_world: float, z_film: tp.Callable, z_world: tp.Callable, c_film: float = None, c_world: float = None):
    '''
    Constructs the geometry of one lens element (two refractive surfaces) in
    an optical system. This comprises two meshes: one describing the lens element, and
    another "baffle"/"aperture" mesh to block rays which do not pass through the lens.

    Inputs: 
    - N: int. The mesh refinement level.
    - r_film: float. The radial extent of the film-side element (closer to -z).
    - r_world: float. The radial extent of the world-side element (closer to +z).
    - z_film: Function(x, y) -> z. The sag function of the film-side element.
    - z_world: Function(x, y) -> z. The sag function of the world-side element.
    - c_film: float (optional). The curvature of the film-side element at z = 0.
    - c_world: float (optional). The curvature of the world-side element at z = 0.

    Outputs:
    - V_out, F_out. Mesh data for the lens element.
    - V_ap, F_ap. Mesh data for the aperture/baffle surrounding the lens element.
    - back_mask, front_mask: np.ndarray[bool]. Mask arrays for accessing/modifying the
        positions of the film-/world-side vertex positions.
    '''
    # =======================================================
    # build hemispheric mesh template
    # =======================================================
    # TODO: hemisphere is not the best choice when lens only subtends a small fraction of the total curvature radius
    # V, F = get_hemisphere(N)

    def draw_quad(v0, v1, v2, v3, flip=False):
        f1 = [v0, v1, v2]
        f2 = [v2, v3, v0]
        if flip:
            f1[0], f1[1] = f1[1], f1[0]
            f2[0], f2[1] = f2[1], f2[0]
        return [f1, f2]

    # TODO
    if c_film is None:
        h = 1e-2 * r_film
        c_film = np.abs(z_film(h, h) - z_film(0, 0)).item() / (h ** 2)
    if c_world is None:
        h = 1e-2 * r_world
        c_world = np.abs(z_world(h, h) - z_world(0, 0)).item() / (h ** 2)

    # =======================================================
    # build lens element geometry
    # =======================================================
    # back side: scale hemisphere by back element radius, apply the sag function z_film(x,y)
    V_back, F = get_spherical_cap(N, r_film, c_film)
    # V_back = V.copy() * r_film
    V_back[:,2] = z_film(V_back[:,0], V_back[:,1])
    F_back = F.copy()
    # invert sense of F_back to flip normals
    F_back[:,[0,1]] = F_back[:,[1,0]]

    Nv, Nf = V_back.shape[0], F_back.shape[0]

    # front side: scale hemisphere by front element radius, apply the sag function z_world(x,y)
    V_front, _ = get_spherical_cap(N, r_world, c_world)
    # V_front = V.copy() * r_world
    V_front[:,2] = z_world(V_front[:,0], V_front[:,1])
    F_front = F.copy() + Nv

    # sew the two pieces together along the unclosed sides of the hemispheres
    open_vertices = boundary_loops(F)[0]
    open_vertices_back = open_vertices.copy()
    open_vertices_front = open_vertices.copy() + Nv
    Nvo = len(open_vertices)
    Fs = []
    for idx in range(Nvo):
        next_idx = (idx + 1) if (idx + 1 < Nvo) else 0
        Fs += draw_quad(
            open_vertices_back[idx], open_vertices_back[next_idx],
            open_vertices_front[next_idx], open_vertices_front[idx])
    Fs = np.array(Fs)

    V_out = np.vstack((V_back, V_front))
    F_out = np.vstack((F_back, F_front, Fs))

    # mask for updating front and back vertices respectively
    # NOTE: allows us to modify sag profile (face curvature, etc.) but NOT the element radius
    back_mask  = np.concatenate((np.ones(Nv),  np.zeros(Nv))).astype(bool)
    front_mask = np.concatenate((np.zeros(Nv), np.ones(Nv))).astype(bool)

    # =======================================================
    # build aperture mesh to occlude rays beyond the back and front elements' extents
    # =======================================================

    def clip_to_box(V, xmax):
        V_ = V.copy()
        V_[:,:2] *= xmax / np.linalg.norm(V[:,:2], axis=1, ord=np.inf)[:,None]
        return V_

    V_back_el = V_back[open_vertices]
    V_back_ap = clip_to_box(V_back_el, 5.0)
    V_front_el = V_front[open_vertices]
    V_front_ap = clip_to_box(V_front_el, 5.0)

    Fs = []
    for idx in range(Nvo):
        next_idx = (idx + 1) if (idx + 1 < Nvo) else 0
        # back_ap -> back_el
        Fs += draw_quad(
            idx, next_idx,                  # V_back_ap[idx], V_back_ap[next_idx],
            Nvo + next_idx, Nvo + idx,      # V_back_el[next_idx], V_back_el[idx], 
            True)
        # back_el -> front_el
        Fs += draw_quad(
            Nvo + idx, Nvo + next_idx,          # V_back_el[idx], V_back_el[next_idx],
            2 * Nvo + next_idx, 2 * Nvo + idx,  # V_front_el[next_idx], V_front_el[idx], 
            True)
        # front_el -> front_ap
        Fs += draw_quad(
            2 * Nvo + idx, 2 * Nvo + next_idx,  # V_front_el[idx], V_front_el[next_idx],  
            3 * Nvo + next_idx, 3 * Nvo + idx,  # V_front_ap[next_idx], V_front_ap[idx], 
            True)
        # front_ap -> back_ap
        Fs += draw_quad(
            3 * Nvo + idx, 3 * Nvo + next_idx,  # V_front_ap[idx], V_front_ap[next_idx],
            next_idx, idx,                      # V_back_ap[next_idx], V_back_ap[idx], 
            True)

    V_ap = np.vstack((V_back_ap, V_back_el, V_front_el, V_front_ap))
    F_ap = np.array(Fs, dtype=np.int32)

    return V_out, F_out, V_ap, F_ap, back_mask, front_mask


def meshplot_element(V_el, F_el, V_ap, F_ap):
    p_ = mp.plot(V_el, F_el, shading={"wireframe": True, "side": "FrontSide"})
    # p_.add_points(V_el, shading={"point_size": 0.3, "point_color": "black"})
    p_.add_mesh(V_ap, F_ap, c=np.array([0,1,0]), shading={"wireframe": True, "side": "FrontSide", "face_color": "green"})
    meshplot_gizmo(p_)
    return p_

def meshplot_gizmo(mplot):
    mplot.add_lines(np.zeros(3), np.array([2,0,0]), shading={"line_color": "blue"})
    mplot.add_lines(np.zeros(3), np.array([0,2,0]), shading={"line_color": "green"})
    mplot.add_lines(np.zeros(3), np.array([0,0,2]), shading={"line_color": "red"})





def create_surface_geometry(
        N: int, 
        r_element: float, 
        compute_z: tp.Callable, 
        c: float = None, 
        baffle_radius: float = 5.0, 
        flip_normals: bool = False):
    '''
    Constructs the geometry of one lens surface in an optical system. This comprises two 
    meshes: one describing the surface element itself, and another "baffle"/"aperture" 
    mesh to block rays whose intersections lie outside the radial extent of the lens.

    Inputs: 
    - N: int. The mesh refinement level.
    - r_element: float. The radial extent of the element.
    - compute_z: Function(x, y) -> z. The sag function of the element surface.
    - c: float (optional). The curvature of the element evaluated at z = 0.
    - baffle_radius: float (optional). The maximum radius of the baffle. Default: 5.0.

    Outputs:
    - V_lens, F_lens. Mesh data for the lens element.
    - V_ap, F_ap. Mesh data for the aperture/baffle surrounding the lens element.
    - ap_mask: np.ndarray[bool]. Mask arrays for accessing/modifying the
        positions of the baffle vertex positions.
    '''
    # =======================================================
    # build hemispheric mesh template
    # =======================================================
    def draw_quad(v0, v1, v2, v3, flip=False):
        f1 = [v0, v1, v2]
        f2 = [v2, v3, v0]
        if flip:
            f1[0], f1[1] = f1[1], f1[0]
            f2[0], f2[1] = f2[1], f2[0]
        return [f1, f2]

    if c is None:
        h = 1e-2 * r_element
        c = np.abs(compute_z(h, h) - compute_z(0, 0)).item() / (h ** 2)

    # =======================================================
    # build lens element geometry
    # =======================================================

    # NOTE: by default, the normals of the surface are always pointing towards +z. 
    # Be sure to set the materials accordingly!
    
    # front side: scale hemisphere by element radius, and then apply the sag 
    # function `compute_z(x,y)`
    V_lens, F_lens = get_spherical_cap(N, r_element, c)
    if flip_normals:
        F_lens[:,[0,1]] = F_lens[:,[1,0]]

    V_lens[:,2] = compute_z(V_lens[:,0], V_lens[:,1])
    Nv, Nf = V_lens.shape[0], F_lens.shape[0]

    # find the vertex/edge loop for the unclosed side of the hemisphere
    open_vertices = boundary_loops(F_lens)[0]
    Nvo = len(open_vertices)

    # mask for accessing the open vertices
    lens_ov_mask = np.zeros(Nv).astype(bool)
    lens_ov_mask[open_vertices] = True

    # =======================================================
    # build aperture mesh to occlude rays beyond the element's extents
    # =======================================================

    def clip_to_box(V, xmax):
        V_ = V.copy()
        V_[:,:2] *= xmax / np.linalg.norm(V[:,:2], axis=1, ord=np.inf)[:,None]
        return V_

    V_ap_el = V_lens[open_vertices]
    V_ap_edge = clip_to_box(V_ap_el, baffle_radius)

    F_ap = []
    # look through the open vertices at `r = r_elem` in a CW/CCW direction
    for idx in range(Nvo):
        next_idx = (idx + 1) if (idx + 1 < Nvo) else 0
        # ap_edge -> ap_el
        F_ap += draw_quad(
            idx, next_idx,                  # V_ap_edge[idx],    V_ap_edge[next_idx]
            Nvo + next_idx, Nvo + idx,      # V_ap_el[next_idx], V_ap_el[idx]
            True)
        # ap_el -> ap_edge
        F_ap += draw_quad(
            Nvo + idx, Nvo + next_idx,      # V_ap_el[idx],        V_ap_el[next_idx]
            next_idx,idx,                   # V_ap_edge[next_idx], V_ap_edge[idx]
            True)

    V_ap = np.vstack((V_ap_el, V_ap_edge))
    F_ap = np.array(F_ap, dtype=np.int32)

    return V_lens, F_lens, V_ap, F_ap, lens_ov_mask
