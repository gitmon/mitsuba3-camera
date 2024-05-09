'''
NOTE: all units are in millimeters!
TODO's: 
    - 2d visualization might be nice (need intersect() code for `Surface`)
    - optimization: per-variable masks and learning rates
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
from .geometry import create_surface_geometry
import meshplot as mp
from igl import read_triangle_mesh
from os.path import join

baffle_radius = 5.0

class Surface:
    def __init__(self, params: dict):
        self.params = params

    def get_params(self):
        return self.params

    def compute_z_dr(self, x, y):
        raise NotImplementedError()

    def compute_z_np(self, x, y):
        raise NotImplementedError()


class ConicSurface(Surface):
    def __init__(self, c: float, K: float, z0: float):
        params = {
            'c': c,
            'K': K,
            'z0': z0,
        }
        super().__init__(params)

    def compute_z_dr(self, x, y):
        ''' 
        Compute the sag function (z-coord) of a conic surface at the radial 
        coordinate r ** 2 = x ** 2 + y ** 2. The resultant z-value is expressed in 
        camera coordinates (i.e. z = 0 is the film; +z faces towards object space).

        This version of the function consumes and outputs `mi.Float` arrays.
        '''
        r2 = dr.sqr(x) + dr.sqr(y)
        safe_sqr = dr.clamp(1 - (1 + self.params['K']) * dr.sqr(self.params['c']) * r2, 0.0, dr.inf)
        z = self.params['z0'] - r2 * self.params['c'] * dr.rcp(1 + dr.sqrt(safe_sqr))
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


class EvenAsphericSurface(Surface):
    def __init__(self, 
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
        params = {
            'c'  : c,
            'K'  : K,
            'z0' : z0,
            'a4' : a4,
            'a6' : a6,
            'a8' : a8,
            'a10': a10,
            'a12': a12,
            'a14': a14,
            'a16': a16,
        }
        super().__init__(params)

    def compute_z_dr(self, x, y):
        ''' 
        Compute the sag function (z-coord) of a conic surface at the radial 
        coordinate r ** 2 = x ** 2 + y ** 2. The resultant z-value is expressed in 
        camera coordinates (i.e. z = 0 is the film; +z faces towards object space).

        This version of the function consumes and outputs `mi.Float` arrays.
        '''
        r2 = dr.sqr(x) + dr.sqr(y)
        z = self.params['a16']
        z = dr.fma(z, r2, self.params['a14'])
        z = dr.fma(z, r2, self.params['a12'])
        z = dr.fma(z, r2, self.params['a10'])
        z = dr.fma(z, r2, self.params['a8'])
        z = dr.fma(z, r2, self.params['a6'])
        z = dr.fma(z, r2, self.params['a4'])

        safe_sqr = dr.clamp(1 - (1 + self.params['K']) * dr.sqr(self.params['c']) * r2, 0.0, dr.inf)
        z = dr.fma(z, r2, self.params['c'] * dr.rcp(1 + dr.sqrt(safe_sqr)))
        # z = self.params['z0'] - r2 * z
        z = dr.fma(z, -r2, self.params['z0'])
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
        self.active_optvars = [key for key, _ in self.params.items()]


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
                # TODO: add ability to select only some variables to optimize
    
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
        # element sizes
        radial_extent: float,
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
        self.radial_extent  = radial_extent
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
            r_element = self.radial_extent,
            compute_z = self.surface.compute_z_np,
            c = self.surface.params['c'],
            flip_normals = not(self.is_world_facing),
            baffle_radius = baffle_radius,
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
                # TODO: add ability to select only some variables to optimize
    
            # TODO: handle per-variable learning rates

        # BSDF/material handling
        # NOTE: no-op here; materials registration is performed in LensSystem!


    def save_init_state(self, params: mi.SceneParameters):
        '''
        Must run this after the lens is added to the scene and the `scene` object is initialized.
        '''
        self.initial_lens_vertices = dr.unravel(mi.Point3f, params[f'{self.lens_key}.vertex_positions'])
        self.initial_baffle_vertices = dr.unravel(mi.Point3f, params[f'{self.baffle_key}.vertex_positions'])


    def update(self, params: mi.SceneParameters, optimizer: mi.ad.optimizers.Optimizer) -> None:
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
            self.radial_extent, 
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
            pass

    def remove_optvars(self, params_to_disable: tp.List[str]):
        active_optvars = [var for var in self.active_optvars if var not in params_to_disable]
        self.active_optvars = active_optvars

    def remove_all_optvars(self):
        self.active_optvars = []



class ApertureElement:
    def __init__(self, 
        element_id: int, 
        # element sizes
        radial_extent: float,
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
        self.radial_extent  = radial_extent
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
            r_element = self.radial_extent,
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
            self.radial_extent, 
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


class LensSystem:
    def __init__(self, 
                 surfaces: tp.List[Surface], 
                 radial_extents: tp.List[float],
                 materials: tp.List [LensMaterial],
                 ):

        if not (len(radial_extents) == len(surfaces)):
            raise AssertionError(f"Radii and surface lists do not match: {len(radial_extents)=}, {len(surfaces)=}")
        
        if not (len(materials) == len(surfaces) - 1):
            raise AssertionError(f"Material and surface lists do not match: {len(materials)=}, {len(surfaces)=}")
        
        # initialize materials. From film->world, the first material in the lens 
        # system is always "air". 
        air_material = LensMaterial()
        materials = [air_material] + materials
        num_materials = len(materials)
        # for mat in materials:
        #     print(mat)

        #
        elements = []
        for idx in range(len(surfaces)):
            # the element's int/ext materials are set by looking at the current and next
            # materials in the list. For the last element, we wrap the `next` material back
            # to material[0] (air).
            next_mat_idx = (idx + 1) % num_materials
            elem = LensElement(
                N=7,
                element_id = len(elements),
                radial_extent = radial_extents[idx],
                surface = surfaces[idx],
                ext_material=materials[idx],
                int_material=materials[next_mat_idx],
                is_world_facing = False,
            )
            elements.append(elem)

        self.elements = elements
        self.rear_z = elements[0].surface.params['z0']
        self.front_z = elements[-1].surface.params['z0']
        self.front_radial_extent = radial_extents[-1]
        self.materials = materials

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
        for element in self.elements:
            element.add_to_scene(scene_dict)

        # add a box around the lens elements and baffles
        zmax = self.front_z * 1.1
        rmax = baffle_radius
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
