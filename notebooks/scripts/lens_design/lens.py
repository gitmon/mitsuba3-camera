'''
NOTE: all units are in millimeters!
TODO's: 
    - optimization: per-variable learning rates
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

import sys
sys.path.insert(0, '/home/chuah/mitsuba3-camera/build/python')
sys.path.append("..")
import drjit as dr
import mitsuba as mi

import typing as tp
from ..mi_utils import create_mesh
from ..plot_utils import plot_cross_section_2d
from .geometry import create_surface_geometry
import meshplot as mp
from igl import read_triangle_mesh
from os.path import join

baffle_radius = 5.0

class Surface:
    def __init__(self, radial_extent: float, params: dict):
        self.params = params
        self.radial_extent = radial_extent

    def get_params(self):
        return self.params

    def compute_z_dr(self, x, y):
        raise NotImplementedError()

    def compute_z_np(self, x, y):
        raise NotImplementedError()
    
    def get_curvature(self):
        '''
        Get the curvature of the element expressed in dimensional units (1/mm).
        '''
        return self.params['c'] / self.radial_extent    # unitless quantity divided by length
    
    def get_z0(self):
        return self.params['z0']
    
    def get_radial_extent(self):
        return self.radial_extent
    # def intersect(self, ray: mi.Ray3f):
    #     raise NotImplementedError()


class ConicSurface(Surface):
    def __init__(self, radial_extent: float, c: float, K: float, z0: float):
        params = {
            'c': c * radial_extent,     # dimensionless curvature
            'K': K,
            'z0': z0,
        }
        super().__init__(radial_extent, params)

    def compute_z_dr(self, x, y):
        ''' 
        Compute the sag function (z-coord) of a conic surface at the radial 
        coordinate r ** 2 = x ** 2 + y ** 2. The resultant z-value is expressed in 
        camera coordinates (i.e. z = 0 is the film; +z faces towards object space).

        This version of the function consumes and outputs `mi.Float` arrays.
        '''
        # dimensional version
        # r2 = dr.sqr(x) + dr.sqr(y)
        # safe_sqr = dr.clamp(1 - (1 + self.params['K']) * dr.sqr(self.params['c']) * r2, 0.0, dr.inf)
        # z = self.params['z0'] - r2 * self.params['c'] * dr.rcp(1 + dr.sqrt(safe_sqr))

        # dimensionless version
        r2 = dr.sqr(x) + dr.sqr(y)
        r2 *= dr.rcp(dr.sqr(self.radial_extent))
        safe_sqr = dr.clamp(1 - (1 + self.params['K']) * dr.sqr(self.params['c']) * r2, 0.0, dr.inf)
        z = self.params['z0'] - self.radial_extent * r2 * self.params['c'] * dr.rcp(1 + dr.sqrt(safe_sqr))
        return z

    def compute_z_np(self, x_, y_):
        ''' 
        Compute the sag function (z-coord) of a conic surface at the radial 
        coordinate r ** 2 = x ** 2 + y ** 2. The resultant z-value is expressed in 
        camera coordinates (i.e. z = 0 is the film; +z faces towards object space).

        This version of the function consumes and outputs numpy arrays.
        '''
        x = mi.Float(x_)
        y = mi.Float(y_)
        return self.compute_z_dr(x, y).numpy()
    
    # def intersect(self, ray: mi.Ray3f):
    #     c = self.params['c']
    #     K = self.params['K']

    #     d = ray.d
    #     o = ray.o - mi.Point3f(0.0, 0.0, self.params['z0'])

    #     A = c * (1.0 + K * dr.sqr(d.z))
    #     B = 2.0 * (c * (dr.dot(o, d) + K * o.z * d.z) - d.z)
    #     C = c * (dr.squared_norm(o) + K * dr.sqr(o.z)) - 2.0 * o.z

    #     Delta = dr.sqr(B) - 4 * A * C
    #     valid = Delta > 0.0

    #     # if Delta < 0.0:
    #     #     return self.dtype(np.inf)

    #     t0 = (-B - dr.sqrt(Delta)) * dr.rcp(2 * A)
    #     t1 = (-B + dr.sqrt(Delta)) * dr.rcp(2 * A)

    #     # if t0 > 0:
    #     #     return t0
    #     # else:
    #     #     return t1

    #     t = dr.select(valid, t0, dr.inf(mi.Float))
        

    # def intersect_conic_newton(self, surf, tol=1e-8, num_iter=10):
    #     # intersect with conic
    #     t = self.intersect_conic(surf)

    #     if np.isinf(t):
    #         return self.dtype(np.inf)

    #     p_conic = self.at(t)
    #     r_conic = np.sqrt(p_conic[0] ** 2 + p_conic[1] ** 2)
    #     o = self.o - surf.origin
    #     d = self.d

    #     p_curr = p_conic
    #     r_curr = r_conic
    #     err = np.inf
    #     itr = 0

    #     while err > tol and itr < num_iter:
    #         # build tangent plane on asphere
    #         z_asph = surf.z_horner(r_curr)
    #         p_asph = np.array([p_curr[0], p_curr[1], z_asph])   
    #         grad = surf.z_full_grad(r_curr)
    #         nx = p_curr[0] / r_curr
    #         ny = p_curr[1] / r_curr
    #         n_asph = np.array([-grad * nx, -grad * ny, 1.0])
    #         n_asph /= np.linalg.norm(n_asph)

    #         # compute intersection with tangent plane
    #         t = np.sum(n_asph * (p_asph - o)) / np.sum(n_asph * d)
    #         p_curr = self.at(t)
    #         r_curr = np.sqrt(p_curr[0] ** 2 + p_curr[1] ** 2)

    #         err = abs(p_curr[2] - z_asph)
    #         itr += 1

    #     print(err, itr)
        
    #     return t

class EvenAsphericSurface(Surface):
    def __init__(self, 
                 radial_extent: float,
                 c:   float, 
                 K:   float, 
                 z0:  float, 
                 a4:  float = 0.0,
                 a6:  float = 0.0,
                 a8:  float = 0.0,
                 a10: float = 0.0,
                 a12: float = 0.0,
                 a14: float = 0.0,
                 a16: float = 0.0,
                 ):
        # non-dimensionalized parameters
        params = {
            'c'  : c * radial_extent,
            'K'  : K,
            'z0' : z0,
            'a4' : a4  * radial_extent ** 3,
            'a6' : a6  * radial_extent ** 5,
            'a8' : a8  * radial_extent ** 7,
            'a10': a10 * radial_extent ** 9,
            'a12': a12 * radial_extent ** 11,
            'a14': a14 * radial_extent ** 13,
            'a16': a16 * radial_extent ** 15,
        }
        super().__init__(radial_extent, params)

    def compute_z_dr(self, x, y):
        ''' 
        Compute the sag function (z-coord) of a conic surface at the radial 
        coordinate r ** 2 = x ** 2 + y ** 2. The resultant z-value is expressed in 
        camera coordinates (i.e. z = 0 is the film; +z faces towards object space).

        This version of the function consumes and outputs `mi.Float` arrays.
        '''
        r2 = dr.sqr(x) + dr.sqr(y)
        r2 *= dr.rcp(dr.sqr(self.radial_extent))
        z = self.params['a16']
        z = dr.fma(z, r2, self.params['a14'])
        z = dr.fma(z, r2, self.params['a12'])
        z = dr.fma(z, r2, self.params['a10'])
        z = dr.fma(z, r2, self.params['a8'])
        z = dr.fma(z, r2, self.params['a6'])
        z = dr.fma(z, r2, self.params['a4'])
        
        # # dimensional version
        # safe_sqr = dr.clamp(1 - (1 + self.params['K']) * dr.sqr(self.params['c']) * r2, 0.0, dr.inf)
        # z = dr.fma(z, r2, self.params['c'] * dr.rcp(1 + dr.sqrt(safe_sqr)))
        # z = dr.fma(z, -r2, self.params['z0'])

        # dimensionless version
        safe_sqr = dr.clamp(1 - (1 + self.params['K']) * dr.sqr(self.params['c']) * r2, 0.0, dr.inf)
        z = dr.fma(z, r2, self.params['c'] * dr.rcp(1 + dr.sqrt(safe_sqr)))
        z = dr.fma(z, -self.radial_extent * r2, self.params['z0'])
        return z

    def compute_z_np(self, x_, y_):
        ''' 
        Compute the sag function (z-coord) of a conic surface at the radial 
        coordinate r ** 2 = x ** 2 + y ** 2. The resultant z-value is expressed in 
        camera coordinates (i.e. z = 0 is the film; +z faces towards object space).

        This version of the function consumes and outputs numpy arrays.
        '''
        x = mi.Float(x_)
        y = mi.Float(y_)
        return self.compute_z_dr(x, y).numpy()


class LensMaterial:
    '''
    TODO: ...

    By default, LensMaterial with an empty constructor produces an "air" material. This 
    instantiation of LensMaterial is treated specially: we cannot add it to the optimizer,
    nor can we update its material properties post-initialization.
    '''
    def __init__(self, 
        name: str = "air", 
        refractive_index: float = 1.000277,
        abbe_number: float = 0.0,
        ):

        self.name = name.lower()
        self.params = {
            'ior': refractive_index,
            'V_d': abbe_number,
            # 'ior': mi.ScalarFloat(refractive_index),
            # 'V_d': mi.ScalarFloat(abbe_number),
        }
        self.param_keys_to_opt_keys = None

        if name == "air":
            self.active_optvars = []
        else:
            self.active_optvars = [key for key, _ in self.params.items()]

    def remove_optvar(self, param_name: str):
        try:
            self.active_optvars.remove(param_name)
        except ValueError:
            print(f"Warning: {param_name} was not found in optvar list!")


    def remove_optvars(self, params_to_disable: tp.List[str]):
        active_optvars = [var for var in self.active_optvars if var not in params_to_disable]
        self.active_optvars = active_optvars


    def remove_all_optvars(self):
        self.active_optvars = []


    def add_to_optimizer(self, optimizer: mi.ad.Optimizer) -> None:
        '''
        Register the material's parameters in the optimizer.
        '''
        if self.name == "air":
            return
        
        if self.param_keys_to_opt_keys is None:
            self.param_keys_to_opt_keys = {}

        # iterate through all the material params
        for param_name, param_value in self.params.items():

            # if this parameter is disabled for optimization, skip adding it to `opt`
            if param_name not in self.active_optvars:
                continue

            optvar_key = f'mat_{self.name}_{param_name}'
            if optvar_key in optimizer:
                raise KeyError(f"Variable {optvar_key} already exists in optimizer!")
            else:
                optimizer[optvar_key] = mi.Float(param_value)
                self.param_keys_to_opt_keys[param_name] = optvar_key
            # TODO: handle per-variable learning rates


    def update(self, optimizer: mi.ad.optimizers.Optimizer) -> None:
        '''
        Update lens material with the new values of the optimized variables.
        '''
        if self.name == "air":
            return
        
        if self.param_keys_to_opt_keys is None:
            return

        # for params that are present in the optimizer, overwrite the old values with the optimizer's values
        for var_key, optvar_key in self.param_keys_to_opt_keys.items():
            self.params[var_key] = optimizer[optvar_key]

    def __str__(self):
        out = f"LensMaterial[\n\tname={self.name},\n\tior={self.params['ior']},\n\tVd={self.params['V_d']},\n]"
        return out



class LensElement:
    def __init__(self, 
        element_id: int, 
        # shape parameters
        surface:  Surface,
        # material parameters
        int_material: LensMaterial,
        ext_material: LensMaterial,
        # meshing parameters
        N: int = 5, 
        is_world_facing: bool = True,
        ):
        '''
        NOTE: the element's BSDF defines an *interface* between two refractive mediums. 
        Thus, need pointers to the media/materials themselves, where the refractive 
        properties are actually controlled.
        '''

        self.subdiv_level = N
        self.id = element_id
        self.int_material = int_material
        self.ext_material = ext_material
        self.surface = surface
        self.param_keys_to_opt_keys = None
        self.lens_fname = None
        self.baffle_fname = None
        self.is_world_facing = is_world_facing
        self.active_optvars = [key for key, _ in self.surface.params.items()]


    def initialize_geometry(self, output_dir: str) -> None:
        '''
        Create the lens geometry and save it to `output_dir`.
        '''
        V_lens, F_lens, V_ap, F_ap, ovs_mask = create_surface_geometry(
            N = self.subdiv_level,
            r_element = self.surface.radial_extent,
            compute_z = self.surface.compute_z_np,
            c = self.surface.get_curvature(),
            flip_normals = not(self.is_world_facing),
            baffle_radius = 1.1 * self.surface.radial_extent,
        )
        
        lens_mesh   = create_mesh(V_lens, F_lens, f"lens{self.id}")
        baffle_mesh = create_mesh(V_ap, F_ap, f"baffle{self.id}")
        ovs_mask   = mi.Mask(ovs_mask)

        lens_fname   = join(output_dir,   f'lens{self.id}.ply')
        baffle_fname = join(output_dir, f'baffle{self.id}.ply')
        lens_mesh  .write_ply(lens_fname)
        baffle_mesh.write_ply(baffle_fname)
        print('[+] Wrote lens mesh (subdivs={}) file to: {}'.format(self.subdiv_level, lens_fname))
        print('[+] Wrote baffle mesh file to: {}'.format(baffle_fname))
        
        self.ovs_mask  = ovs_mask

        if (self.lens_fname is None) or (self.baffle_fname is None):
            self.lens_fname = lens_fname
            self.baffle_fname = baffle_fname


    def meshplot_geometry(self, p_ = None, lens_c = np.array([0,1,1]), baffle_c = np.array([0,1,0]), **kwargs) -> None:
        '''
        Visualize the lens geometry using Meshplot.
        '''
        if p_ is None:
            p_ = mp.plot(*read_triangle_mesh(self.lens_fname), c=lens_c, **kwargs)
        else:
            p_ .add_mesh(*read_triangle_mesh(self.lens_fname), c=lens_c, **kwargs)
            
        p_.add_mesh(*read_triangle_mesh(self.baffle_fname), c=baffle_c, **kwargs)

        return p_


    def add_to_scene(self, scene_dict) -> None:
        '''
        Register the lens element in the scene dictionary.
        '''
        lens_key = f"lens{self.id}"
        baffle_key = f"baffle{self.id}"
        bsdf_key = f"bsdf{self.id}_{self.ext_material.name}-to-{self.int_material.name}"

        bsdf_dict = {
                'type': 'dispersive',
                'id': bsdf_key,
                'ext_ior': self.ext_material.params['ior'],
                'ext_V_d': self.ext_material.params['V_d'],
                'int_ior': self.int_material.params['ior'],
                'int_V_d': self.int_material.params['V_d'],
                'specular_reflectance': { 'type': 'spectrum', 'value': 0 },
            }

        lens_dict = {
                'type': 'ply',
                'id': lens_key,
                'filename': self.lens_fname,
                # 'bsdf': {'type': 'ref', 'id': 'simple-glass'},
                'bsdf': bsdf_dict,
            }

        if 'black-bsdf' not in scene_dict:
            scene_dict['black-bsdf'] = {
                'type': 'diffuse',
                'id': 'black-bsdf',
                'reflectance': { 'type': 'spectrum', 'value': 0 },
            }

        baffle_dict = {
                'type': 'ply',
                'id': baffle_key,
                'filename': self.baffle_fname,
                'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
            }

        if lens_key not in scene_dict:
            scene_dict[lens_key] = lens_dict
            # print(lens_dict)
        else:
            raise KeyError(f"Lens `{lens_key}` already exists in scene!")

        if baffle_key not in scene_dict:
            scene_dict[baffle_key] = baffle_dict
        else:
            raise KeyError(f"Baffle `{baffle_key}` already exists in scene!")

        self.lens_key = lens_key
        self.baffle_key = baffle_key


    def add_to_optimizer(self, optimizer: mi.ad.Optimizer) -> None:
        '''
        Register the shape parameters for this lens in the optimizer.
        '''
        if self.param_keys_to_opt_keys is None:
            self.param_keys_to_opt_keys = {}

        # iterate through *all* the shape params
        # self.surface.params = {'c': 1.0, 'K', 1.0, ...}
        for var_name, value in self.surface.params.items():

            # if this parameter is disabled for optimization, skip adding it to `opt`
            if var_name not in self.active_optvars:
                continue

            optvar_key = f'lens{self.id}_{var_name}'
            if optvar_key in optimizer:
                raise KeyError(f"Variable {optvar_key} already exists in optimizer!")
            else:
                optimizer[optvar_key] = mi.Float(value)
                self.param_keys_to_opt_keys[var_name] = optvar_key
            # TODO: handle per-variable learning rates

        # BSDF/material handling
        # NOTE: no-op here; materials registration is performed in LensSystem!


    def save_init_state(self, params: mi.SceneParameters):
        '''
        Must run this after the lens is added to the scene and the `scene` object is initialized.
        '''
        self.initial_lens_vertices = dr.unravel(mi.Point3f, params[f'{self.lens_key}.vertex_positions'])
        self.initial_baffle_vertices = dr.unravel(mi.Point3f, params[f'{self.baffle_key}.vertex_positions'])


    def update(self, params: mi.SceneParameters, optimizer: mi.ad.optimizers.Optimizer, ext_params: dict = None) -> None:
        '''
        Update lens element with the new values of the optimized variables, and recompute the lens geometry.
        '''
        # BSDF/materials handling. 
        # NOTE: materials update is performed in LensSystem! Here, we simply need to copy
        # the materials' updated values into the lens BSDF
        # NOTE: materials update *must* be performed before LensElement update

        params[f'{self.lens_key}.bsdf.int_ior_d'] = self.int_material.params['ior']
        params[f'{self.lens_key}.bsdf.int_V_d']   = self.int_material.params['V_d']
        params[f'{self.lens_key}.bsdf.ext_ior_d'] = self.ext_material.params['ior']
        params[f'{self.lens_key}.bsdf.ext_V_d']   = self.ext_material.params['V_d']

        if self.param_keys_to_opt_keys is None:
            return

        # Compute new vertex positions for the surface
        # first, load the existing shape params
        new_shape_params = self.surface.params

        # for params that are present in the optimizer, overwrite the old values with the optimizer's values
        for var_key, optvar_key in self.param_keys_to_opt_keys.items():
            new_shape_params[var_key] = optimizer[optvar_key]

        # allow some shape params to be set directly (e.g. to implement the constrained rear surface)
        if ext_params is not None:
            for param_key, param_value in ext_params.items():
                new_shape_params[param_key] = param_value

        new_vertex_pos = mi.Point3f(
                self.initial_lens_vertices[0], 
                self.initial_lens_vertices[1], 
                self.surface.compute_z_dr(
                    self.initial_lens_vertices[0], 
                    self.initial_lens_vertices[1]))

        # Flatten the vertex position array before assigning it to `params`
        params[f'{self.lens_key}.vertex_positions'] = dr.ravel(new_vertex_pos)

        # Next, update the baffle's vertex positions. In practice, only their z-positions
        # are able to change; they should be updated to match the open boundary of the lens 
        # surface whenever the latter's axial position is modified

        lens_ovs_z = self.surface.compute_z_dr(
            self.surface.radial_extent, 
            mi.Float(0.0)) * dr.ones(mi.Float, dr.width(self.initial_baffle_vertices))

        # update the baffle vertices' z-positions
        new_baffle_pos = mi.Point3f(
            self.initial_baffle_vertices[0],
            self.initial_baffle_vertices[1],
            lens_ovs_z,
        )

        # Flatten the vertex position array before assigning it to `params`
        params[f'{self.baffle_key}.vertex_positions'] = dr.ravel(new_baffle_pos)

        # Propagate changes through the scene (e.g. rebuild BVH)
        # NOTE: BVH update is performed in LensSystem

    def remove_optvar(self, param_name: str):
        try:
            self.active_optvars.remove(param_name)
        except ValueError:
            print(f"Warning: {param_name} was not found in optvar list!")

    def remove_optvars(self, params_to_disable: tp.List[str]):
        active_optvars = [var for var in self.active_optvars if var not in params_to_disable]
        self.active_optvars = active_optvars

    def remove_all_optvars(self):
        self.active_optvars = []



class ApertureElement:
    def __init__(self, 
        element_id: int, 
        # shape parameters
        surface:  Surface,
        # meshing parameters
        N: int = 5, 
        is_world_facing: bool = True,
        ):
        '''
        NOTE: the element's BSDF defines an *interface* between two refractive mediums. 
        Thus, need pointers to the media/materials themselves, where the refractive 
        properties are actually controlled.
        '''

        self.subdiv_level = N
        self.id = element_id
        self.surface = surface
        self.param_keys_to_opt_keys = None
        self.ap_fname = None
        self.is_world_facing = is_world_facing
        self.active_optvars = [key for key, _ in self.surface.params.items()]


    def initialize_geometry(self, output_dir: str) -> None:
        '''
        Create the lens geometry and save it to `output_dir`.
        '''
        _, _, V_ap, F_ap, _ = create_surface_geometry(
            N = self.subdiv_level,
            r_element = self.surface.radial_extent,
            compute_z = self.surface.compute_z_np,
            c = 0.0,
            flip_normals = not(self.is_world_facing),
            baffle_radius = baffle_radius,
        )
        
        ap_mesh = create_mesh(V_ap, F_ap, f"baffle{self.id}")

        ap_fname = join(output_dir, f'baffle{self.id}.ply')
        ap_mesh.write_ply(ap_fname)
        print('[+] Wrote aperture mesh file to: {}'.format(ap_fname))
        
        if self.ap_fname is None:
            self.ap_fname = ap_fname


    def meshplot_geometry(self, p_ = None, lens_c = np.array([0,1,1]), **kwargs) -> None:
        '''
        Visualize the lens geometry using Meshplot.
        '''
        if p_ is None:
            p_ = mp.plot(*read_triangle_mesh(self.ap_fname), c=lens_c, **kwargs)
        else:
            p_ .add_mesh(*read_triangle_mesh(self.ap_fname), c=lens_c, **kwargs)
            
        return p_


    def add_to_scene(self, scene_dict) -> None:
        '''
        Register the lens element in the scene dictionary.
        '''
        ap_key = f"lens{self.id}_AP"

        if 'black-bsdf' not in scene_dict:
            scene_dict['black-bsdf'] = {
                'type': 'diffuse',
                'id': 'black-bsdf',
                'reflectance': { 'type': 'spectrum', 'value': 0 },
            }

        ap_dict = {
                'type': 'ply',
                'id': ap_key,
                'filename': self.ap_fname,
                'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
            }

        if ap_key not in scene_dict:
            scene_dict[ap_key] = ap_dict
        else:
            raise KeyError(f"Aperture `{ap_key}` already exists in scene!")

        self.ap_key = ap_key


    def add_to_optimizer(self, optimizer: mi.ad.Optimizer) -> None:
        '''
        Register the shape parameters for this lens in the optimizer.
        '''
        pass

        if self.param_keys_to_opt_keys is None:
            self.param_keys_to_opt_keys = {}

        # iterate through *all* the shape params
        # self.surface.params = {'c': 1.0, 'K', 1.0, ...}
        for var_name, value in self.surface.params.items():

            # if this parameter is disabled for optimization, skip adding it to `opt`
            if var_name not in self.active_optvars:
                continue

            optvar_key = f'ap{self.id}_{var_name}'
            if optvar_key in optimizer:
                raise KeyError(f"Variable {optvar_key} already exists in optimizer!")
            else:
                optimizer[optvar_key] = mi.Float(value)
                self.param_keys_to_opt_keys[var_name] = optvar_key

    def save_init_state(self, params: mi.SceneParameters):
        '''
        Must run this after the lens is added to the scene and the `scene` object is initialized.
        '''
        self.initial_ap_vertices = dr.unravel(mi.Point3f, params[f'{self.ap_key}.vertex_positions'])


    def update(self, params: mi.SceneParameters, optimizer: mi.ad.optimizers.Optimizer) -> None:
        '''
        Update lens element with the new values of the optimized variables, and recompute the lens geometry.
        '''
        if self.param_keys_to_opt_keys is None:
            return

        # Next, update the baffle's vertex positions. In practice, only their z-positions
        # are able to change; they should be updated to match the open boundary of the lens 
        # surface whenever the latter's axial position is modified

        lens_ovs_z = self.surface.compute_z_dr(
            self.surface.radial_extent, 
            mi.Float(0.0)) * dr.ones(mi.Float, dr.width(self.initial_baffle_vertices))

        # update the aperture vertices' z-positions
        new_ap_pos = mi.Point3f(
            self.initial_ap_vertices[0],
            self.initial_ap_vertices[1],
            lens_ovs_z,
        )

        # Flatten the vertex position array before assigning it to `params`
        params[f'{self.ap_key}.vertex_positions'] = dr.ravel(new_ap_pos)

        # Propagate changes through the scene (e.g. rebuild BVH)
        # NOTE: BVH update is performed in LensSystem

    def remove_optvar(self, param_name: str):
        try:
            self.active_optvars.remove(param_name)
        except ValueError:
            print(f"Warning: {param_name} was not found in optvar list!")
            pass

    def remove_optvars(self, params_to_disable: tp.List[str]):
        active_optvars = [var for var in self.active_optvars if var not in params_to_disable]
        self.active_optvars = active_optvars

    def remove_all_optvars(self):
        self.active_optvars = []


def matrix2f_inverse(A: mi.Matrix2f):
    det = dr.rcp(A[0,0] * A[1,1] - A[1,0] * A[0,1])
    return det * mi.Matrix2f(A[1,1], -A[0,1], -A[1,0], A[0,0])


class LensSystem:
    def __init__(self, 
                 surfaces: tp.List[Surface], 
                 materials: tp.List [LensMaterial],
                 aperture_index: int = None,
                 ):

        if not (len(materials) == len(surfaces) - 1):
            raise AssertionError(f"Material and surface lists do not match: {len(materials)=}, {len(surfaces)=}")
        
        if aperture_index is None:
            aperture_index = -1

        # initialize materials. From film->world, the first material in the lens 
        # system is always "air". 
        air_material = LensMaterial()
        materials = [air_material] + materials
        num_materials = len(materials)

        elements = []
        for idx in range(len(surfaces)):
            # the element's int/ext materials are set by looking at the current and next
            # materials in the list. For the last element, we wrap the `next` material back
            # to material[0] (air).
            next_mat_idx = (idx + 1) % num_materials
            elem = LensElement(
                N=7,
                element_id = len(elements),
                surface = surfaces[idx],
                ext_material=materials[idx],
                int_material=materials[next_mat_idx],
                is_world_facing = False,
            )

            elements.append(elem)

        self.elements = elements
        self.rear_z = elements[0].surface.params['z0']
        self.front_z = elements[-1].surface.params['z0']
        self.front_radial_extent = elements[-1].surface.radial_extent
        self.materials = materials
        self.aperture_index = aperture_index

        self.compute_paraxial_quantities()

    def compute_paraxial_quantities(self):
        M_film2front = mi.Matrix2f(1, 0, 0, 1)
        M_front2film = mi.Matrix2f(1, 0, 0, 1)
        M_rear2front = mi.Matrix2f(1, 0, 0, 1)

        # for pupil calculations
        M_film2ap = mi.Matrix2f(1, 0, 0, 1)
        M_ap2front = mi.Matrix2f(1, 0, 0, 1)

        num_materials = len(self.materials)

        for idx in range(len(self.elements)):
            # compute and concatenate film-to-world ray matrices
            next_mat_idx = (idx + 1) % num_materials

            if idx == 0:
                z_prev = 0
                z_curr = self.elements[idx].surface.params['z0']
            else:
                z_prev = self.elements[idx - 1].surface.params['z0']
                z_curr = self.elements[idx].surface.params['z0']
            
            thickness = z_curr - z_prev
            curvature = -self.elements[idx].surface.get_curvature()
            ior_i = self.materials[idx].params['ior']
            ior_f = self.materials[next_mat_idx].params['ior']
            m10 = -(ior_f - ior_i) * curvature / ior_f
            m11 = ior_i / ior_f
            lens_matrix = mi.Matrix2f(1, 0, m10, m11)
            transit_matrix = mi.Matrix2f(1, thickness, 0, 1)
            elem_matrix_fwd = lens_matrix @ transit_matrix
            
            # compute matrix products
            M_film2front = elem_matrix_fwd @ M_film2front

            # pupil matrices
            if idx < self.aperture_index:
                # compute M_rear
                M_film2ap = elem_matrix_fwd @ M_film2ap
            elif idx == self.aperture_index:
                # include surf->AP transit matrix in M_rear
                M_film2ap = transit_matrix @ M_film2ap
                M_ap2front = lens_matrix @ M_ap2front
            else:
                # compute M_front
                M_ap2front = elem_matrix_fwd @ M_ap2front

            # special case for the rear2front matrix: skip film->rear transit
            if idx == 0:
                M_rear2front = lens_matrix @ M_rear2front
            else:
                M_rear2front = elem_matrix_fwd @ M_rear2front


        M_front2film = matrix2f_inverse(M_film2front)

        # for raytracing
        self.ray_matrix_film2front  = M_film2front
        self.ray_matrix_front2film  = M_front2film

        # for EFL, BFL constraints
        self.ray_matrix_rear2front  = M_rear2front

        # for pupil calculations
        self.pupil_rear_matrix = M_film2ap
        self.pupil_front_matrix = M_ap2front
        z_exit, z_entrance, r_exit, r_entrance = self.compute_pupils()
        self.exit_pupil_position = z_exit
        self.exit_pupil_radius = r_exit
        self.entrance_pupil_position = z_entrance
        self.entrance_pupil_radius = r_entrance
        # print(f"{r_exit=}, {r_entrance=}")


    def get_EFL_paraxial_matrix(self):
        ray_matrix_S1_to_front = mi.Matrix2f(1, 0, 0, 1)
        num_materials = len(self.materials)

        for idx in range(len(self.elements)):
            # skip rear element when tabulating matrices
            if idx == 0:
                continue

            next_mat_idx = (idx + 1) % num_materials
            z_prev = self.elements[idx - 1].surface.params['z0']
            z_curr = self.elements[idx].surface.params['z0']
            thickness = z_curr - z_prev
            curvature = -self.elements[idx].surface.get_curvature()
            ior_i = self.materials[idx].params['ior']
            ior_f = self.materials[next_mat_idx].params['ior']
            m10 = -(ior_f - ior_i) * curvature / ior_f
            m11 = ior_i / ior_f
            lens_matrix = mi.Matrix2f(1, 0, m10, m11)
            transit_matrix = mi.Matrix2f(1, thickness, 0, 1)
            elem_matrix_fwd = lens_matrix @ transit_matrix
            
            # skip the transit matrix for the lens after L1
            if idx == 1:
                ray_matrix_S1_to_front = lens_matrix @ ray_matrix_S1_to_front
            else:
                ray_matrix_S1_to_front = elem_matrix_fwd @ ray_matrix_S1_to_front
        
        return ray_matrix_S1_to_front


    def trace_paraxial_film2front(self, ray: mi.Vector2f) -> mi.Vector2f:
        '''
        Traces a paraxial ray (y,u) forwards from the film plane (z = 0) to the
        last element interface (z = front_z). The ray is oriented in the +z
        (film->world) direction.
        '''
        return self.ray_matrix_film2front @ ray

    def trace_paraxial_front2film(self, ray: mi.Vector2f) -> mi.Vector2f:
        '''
        Traces a paraxial ray (y,u) backwards from the last element interface 
        (z = front_z) onto the film plane (z = 0). The ray is oriented in the 
        +z (film->world) direction.
        '''
        return self.ray_matrix_front2film @ ray


    def compute_BFL(self) -> mi.Float:
        '''
        Compute the lens system's back focal length using the paraxial ray matrices. 
        Note that since our lens coordinate system's convention is inverted, the BFL 
        and FFL computed by programs like RayOptics should be swapped to get our 
        system's BFL/FFL.
        '''
        return -self.ray_matrix_rear2front[1,1] / self.ray_matrix_rear2front[1,0]

    def compute_FFL(self) -> mi.Float:
        '''
        Compute the lens system's front focal length using the paraxial ray matrices. 
        Note that since our lens coordinate system's convention is inverted, the BFL 
        and FFL computed by programs like RayOptics should be swapped to get our 
        system's BFL/FFL.
        '''
        return -self.ray_matrix_rear2front[0,0] / self.ray_matrix_rear2front[1,0]

    def compute_EFL(self) -> mi.Float:
        '''
        Compute the lens system's effective focal length using the paraxial ray matrices. 
        '''
        return -1.0 / self.ray_matrix_rear2front[1,0]
    
    def compute_pupils(self):
        '''
        Compute the pupil positions in the paraxial approximation.
        '''
        z_exit = 0 + self.pupil_rear_matrix[0,1] / self.pupil_rear_matrix[0,0]
        z_entrance = self.elements[-1].surface.params['z0'] - \
            self.pupil_front_matrix[0,1] / self.pupil_front_matrix[1,1]
        
        ap_radius = self.elements[self.aperture_index].surface.radial_extent
        r_exit = ap_radius / self.pupil_rear_matrix[0,0]
        r_entrance = ap_radius / self.pupil_front_matrix[1,1]
        
        return z_exit[0], z_entrance[0], r_exit[0], r_entrance[0]
    
    def get_rear_surface_params(self, f):
        '''
        Modify the rear surface's curvature and axial position to i) constrain the lens system's
        overall focal length, and ii) ensure it is focused at the film plane.
        Input: 
            - f: float. Desired effective focal length in mm.
        '''
        Mk = self.get_EFL_paraxial_matrix()
        A, B, C, D = Mk[0,0], Mk[0,1], Mk[1,0], Mk[1,1]
        n1 = self.materials[0].params['ior']
        n2 = self.materials[1].params['ior']
        k = n1 * dr.rcp(n2)
        z2 = self.elements[1].surface.params['z0']

        # focus-on-film-plane condition sets the rear element's distance from the
        # film plane
        surf_z0 = f * k * (C * z2 + D) / ((1.0 + f * k * C))

        # focal length condition sets the curvature of the rear element surface
        surf_c = -(dr.rcp(f) + C) * dr.rcp((C * (z2 - surf_z0) + D) * (1 - k))

        # convert curvature to nondimensional, since we are assigning its value directly
        # to the LensElement.Surface
        radial_extent = self.elements[0].surface.radial_extent
        surf_c *= radial_extent
        surf_params = { 'c': surf_c, 'z0': surf_z0 }

        return surf_params

    def size(self):
        return len(self.elements)

    def save_init_state(self, params: mi.SceneParameters):
        for element in self.elements:
            element.save_init_state(params)

    def update(self, params: mi.SceneParameters, optimizer: mi.ad.Optimizer) -> None:
        for material in self.materials:
            material.update(optimizer)

        for element in self.elements:
            element.update(params, optimizer)

        # update the paraxial matrices after element and material data
        # are updated
        self.compute_paraxial_quantities()

        # Update scene params at the end to rebuild the BVH
        params.update()

    def fixed_EFL_update(self, params: mi.SceneParameters, optimizer: mi.ad.Optimizer, efl: float) -> None:
        for material in self.materials:
            material.update(optimizer)

        if efl is not None:
            for element in self.elements[1:]:
                element.update(params, optimizer)
            # update rear element params
            rear_params = self.get_rear_surface_params(efl)
            self.elements[0].update(params, optimizer, rear_params)
        else:
            for element in self.elements:
                element.update(params, optimizer)

        # update the paraxial matrices after element and material data
        # are updated
        self.compute_paraxial_quantities()

        # Update scene params at the end to rebuild the BVH
        params.update()

    def meshplot_geometry(self, p_ = None, **kwargs):
        '''
        Visualize the lens system's geometry using Meshplot.
        '''

        num_elements = len(self.elements)
        colors = np.array([colormaps['viridis'](x) for x in np.linspace(0, 1, num_elements)])
        # drop alpha channel
        colors = colors[:,:3]
        
        if p_ is None:
            p_ = self.elements[0].meshplot_geometry(lens_c=colors[0], baffle_c=colors[0] * 0.5, **kwargs)
        else:
            p_ = self.elements[0].meshplot_geometry(p_=p_, lens_c=colors[0], baffle_c=colors[0] * 0.5, **kwargs)

        for element in self.elements[1:]:
            c = colors[element.id]
            element.meshplot_geometry(p_=p_, lens_c=c, baffle_c=c * 0.5, **kwargs)

        return p_

    def add_to_scene(self, scene_dict):
        rmax = 0.0
        for element in self.elements:
            element.add_to_scene(scene_dict)
            rmax = max(rmax, element.surface.radial_extent)

        # add a box around the lens elements and baffles
        zmax = self.front_z * 1.1
        rmax = rmax * 1.1
        origins = [
            np.array([ 1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([0,  1, 0]),
            np.array([0, -1, 0]),
        ]
        z_vector = np.array([0,0,1])
        for i, origin in enumerate(origins):
            scene_key = f'camera_housing_{i}'
            box_dict = {
                # 'type': 'rectangle',
                'type': 'obj',
                'id': scene_key,
                'filename': 'meshes/rectangle.obj',
                'to_world': mi.ScalarTransform4f.scale([rmax, -rmax, zmax / 2]) @ 
                            mi.ScalarTransform4f.look_at(
                                origin=z_vector + origin,
                                target=z_vector + origin * 0,
                                up=[0, 0, 1]),
                'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
            }
            scene_dict[scene_key] = box_dict


    def disable_all_surfaces(self):
        '''
        Disable optimization of all lens surfaces' parameters.
        '''
        for element in self.elements:
            element.remove_all_optvars()

    def disable_all_materials(self):
        '''
        Disable optimization of all lens materials' parameters.
        '''
        for material in self.materials:
            material.remove_all_optvars()

    def disable_surface_at_index(self, index: int):
        '''
        Disable optimization of the i-th surface's shape parameters.
        '''
        self.elements[index].remove_all_optvars()

    def disable_material_at_index(self, index: int):
        '''
        Disable optimization of the i-th material's parameters.
        '''
        self.materials[index].remove_all_optvars()

    def disable_surface_vars(self, param_name: str):
        '''
        Disable optimization of the parameter `param_name` in all surfaces.
        '''
        for element in self.elements:
            element.remove_optvar(param_name)

    def disable_material_vars(self, param_name: str):
        '''
        Disable optimization of the parameter `param_name` in all materials.
        '''
        for material in self.materials:
            material.remove_optvar(param_name)

    def disable_var_in_surface(self, param_name: str, index: int):
        '''
        Disable optimization of the parameter `param_name` in the i-th surface.
        '''
        self.elements[index].remove_optvar(param_name)

    def disable_var_in_material(self, param_name: str, index: int):
        '''
        Disable optimization of the parameter `param_name` in the i-th material.
        '''
        self.materials[index].remove_optvar(param_name)

    def add_to_optimizer(self, optimizer):
        for material in self.materials:
                material.add_to_optimizer(optimizer)

        for element in self.elements:
            element.add_to_optimizer(optimizer)


    def initialize_geometry(self, output_dir):
        for element in self.elements:
            element.initialize_geometry(output_dir)


    def draw_cross_section(self, N, label: str = None, color='k', fig = None, **kwargs):
        cs_edges = []
        cs_points = []
        idx_offset = 0
        is_new_elem = False

        def draw_surface(elem: LensElement, outwards=True):
            xs = np.linspace(0, elem.surface.radial_extent, N)
            if not(outwards):
                xs = xs[::-1]
            ys = np.zeros_like(xs)
            points = np.c_[xs, ys, elem.surface.compute_z_np(xs, ys)]
            return points

        for surface_id in range(len(self.elements)):
            element = self.elements[surface_id]
            left_is_air = element.ext_material.name == "air"
            right_is_air = element.ext_material.name == "air"

            is_new_elem = left_is_air
            points_elem = draw_surface(element, is_new_elem)
            # num_points_elem = points_elem.shape[0]
            cs_points.append(points_elem)
            # if is_new_elem and idx_offset > 0:
            #     cs_edges += [[idx_offset - 1, idx_offset]]
            if is_new_elem:
                cs_edges += [[idx_offset + i, idx_offset + i + 1] for i in range(N - 1)]
            else:
                cs_edges += [[idx_offset + i - 1, idx_offset + i] for i in range(N)]
            idx_offset += N
            
            # for glass-glass interfaces, draw the interface a second time
            if not(left_is_air) and not(right_is_air):
                is_new_elem = True
                points_elem = draw_surface(element, is_new_elem)
                # num_points_elem = points_elem.shape[0]
                cs_points.append(points_elem)
                if is_new_elem:
                    cs_edges += [[idx_offset + i, idx_offset + i + 1] for i in range(N - 1)]
                else:
                    cs_edges += [[idx_offset + i - 1, idx_offset + i] for i in range(N)]
                idx_offset += N
            
        cs_points = np.concatenate(cs_points)
        cs_edges = np.array(cs_edges)

        # if label is None:
        #     label = "Lens"
        if fig == None:
            plt.figure()
        plot_cross_section_2d(cs_points, cs_edges, color=color, label=label, **kwargs)
        cs_points[:,0] *= -1
        plot_cross_section_2d(cs_points, cs_edges, color=color, label=label, **kwargs)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')
        return plt.gcf()
    
    def print(self):
        print("\n====== Materials ======")
        for i, mat in enumerate(reversed(self.materials)):
            if mat.name == "air":
                A, B = 1.000277, 0.0
            else:
                A, B = abbe_to_cauchy(mat.params['ior'], mat.params['V_d'])
                A = np.array(A).item()
                B = np.array(B).item()
            print(f"mat[{i}]: {A:.12f}f, {B:.12f}f")

        print("\n====== Geometry ======")
        curvature_radii = [1.0 / e.surface.get_curvature() for e in self.elements]
        z0s = [0] + [e.surface.get_z0() for e in self.elements]
        thicknesses = [z0s[i + 1] - z0s[i] for i in range(len(self.elements))]
        extents = [e.surface.get_radial_extent() for e in self.elements]

        for lst in (curvature_radii, thicknesses, extents):
            for i in range(len(lst)):
                lst[i] = np.array(lst[i]).item()

        print("r_c:", ", ".join(["{:.8f}".format(x) + "f" for x in  curvature_radii[::-1]]))
        print("  t:", ", ".join(["{:.8f}".format(x) + "f" for x in      thicknesses[::-1]]))
        print("r_e:", ", ".join(["{:.8f}".format(x) + "f" for x in          extents[::-1]]))


def abbe_to_cauchy(nd, Vd, lbdas=[0.5893, 0.48613, 0.65627]):
    lbda_D, lbda_F, lbda_C = lbdas
    B = (nd - 1) / (Vd * (lbda_F ** -2 - lbda_C ** -2))
    A = nd - B * lbda_D ** -2
    return A, B


def __eval_focal_lengths_singlet(c0, c1):
    surf0 = { 'radial_extent': 0.8, 'c': c0, 'K': 0.0, 'z0': 2.5 }
    surf1 = { 'radial_extent': 0.8, 'c': c1,  'K': 0.0, 'z0': 2.8 }
    surf0 = ConicSurface(**surf0)
    surf1 = ConicSurface(**surf1)
    surfaces = [surf0, surf1]
    materials = [LensMaterial("nbk7", 1.5047, 64.17)]
    lens_system = LensSystem(surfaces, materials)
    bfl = lens_system.compute_BFL().numpy().item()
    ffl = lens_system.compute_FFL().numpy().item()
    return np.array([bfl, ffl])

def test_focal_lengths_singlet():
    test_datas = [
        {
            'input': {'c0': -0.3, 'c1': 0.3}, 
            'output': np.array([3.25439588, 3.25439588]),
        },
        {
            'input': {'c0': -0.277226358652119, 'c1': 0.421164870262146}, 
            'output': np.array([2.76563327, 2.80743881])
        }]
    
    for data_dict in test_datas:
        assert np.allclose(__eval_focal_lengths_singlet(**data_dict['input']), data_dict['output'])

def __eval_focal_lengths_cooke():
    # NOTE: curvature data are transcribed as radii and inverted later
    surfs = [
        { 'radial_extent': 5.0, 'c': -17.285, 'z0': 42.95 },
        { 'radial_extent': 5.0, 'c': 141.25,  'z0': 44.95 },
        { 'radial_extent': 5.0, 'c': 19.3,    'z0': 50.95 },
        { 'radial_extent': 5.0, 'c': -20.25,  'z0': 51.95 },
        { 'radial_extent': 5.0, 'c': -158.65, 'z0': 57.95 },
        { 'radial_extent': 5.0, 'c': 21.25,   'z0': 59.95 },
    ]
    for surf in surfs:
        surf['c'] = 1.0 / surf['c']
    surfaces = [ConicSurface(K=0, **surf) for surf in surfs]
    air = LensMaterial("air")
    sk16 = LensMaterial("sk16", 1.62041, 60.32)
    f4 = LensMaterial("f4", 1.616592, 36.63)
    materials = [sk16, air, f4, air, sk16]
    lens_system = LensSystem(surfaces, materials)
    efl = lens_system.compute_EFL().numpy().item()
    print("BFL matches image plane: ", lens_system.compute_BFL(), surfs[0]['z0'])
    return np.array([efl])

def __eval_focal_lengths_petzval():
    # NOTE: curvature data are transcribed as radii and inverted later
    surfs = [
        { 'radial_extent': 0.8, 'c': -188.1, 'z0': 20.4 },
        { 'radial_extent': 0.8, 'c': -22.1,  'z0': 22.4 },
        { 'radial_extent': 0.8, 'c': 28.42,  'z0': 28.9 },
        { 'radial_extent': 0.8, 'c': np.inf, 'z0': 64.67 },
        { 'radial_extent': 0.8, 'c': -37.87, 'z0': 66.67 },
        { 'radial_extent': 0.8, 'c': 36.27,  'z0': 72.67 },
    ]
    for surf in surfs:
        surf['c'] = 1.0 / surf['c']
    surfaces = [ConicSurface(K=0, **surf) for surf in surfs]
    air = LensMaterial("air")
    k7 = LensMaterial("k7", 1.51112, 60.41)
    f2 = LensMaterial("f2", 1.62004, 36.37)
    materials = [f2, k7, air, f2, k7]
    lens_system = LensSystem(surfaces, materials)
    efl = lens_system.compute_EFL().numpy().item()
    print("BFL matches image plane: ", lens_system.compute_BFL(), surfs[0]['z0'])
    return np.array([efl])

def test_focal_lengths_compound():
    print(__eval_focal_lengths_petzval(), np.array([49.999855]))
    print(__eval_focal_lengths_cooke(),   np.array([50.000541]))

def test_focal_lengths():
    test_focal_lengths_singlet()
    test_focal_lengths_compound()