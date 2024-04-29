import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/chuah/mitsuba3-camera/build/python')
sys.path.append("..")
import drjit as dr
import mitsuba as mi

import typing as tp
from ..mi_utils import create_mesh
from .geometry import create_lens_geometry
import meshplot as mp
from igl import read_triangle_mesh
from os.path import join

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


class LensElement:
    def __init__(self, 
        element_id: int, 
        # element sizes
        radial_extent_film: float,
        radial_extent_world: float,
        # shape parameters
        surf_film:  Surface,
        surf_world: Surface, 
        # meshing parameters
        N: int = 5, 
        ):

        self.subdiv_level = N
        self.radial_extent_film  = radial_extent_film
        self.radial_extent_world = radial_extent_world
        self.id = element_id
        
        
        self.surf_film = surf_film
        self.surf_world = surf_world
        self.param_keys_to_opt_keys = None
        self.lens_fname = None
        self.baffle_fname = None



    def initialize_geometry(self, output_dir: str) -> None:
        '''
        Create the lens geometry and save it to `output_dir`.
        '''
        V_lens, F_lens, V_ap, F_ap, film_mask, world_mask = create_lens_geometry(
            N = self.subdiv_level,
            r_film  = self.radial_extent_film,
            r_world = self.radial_extent_world,
            z_film  = self.surf_film.compute_z_np,
            z_world = self.surf_world.compute_z_np,
            c_film  = self.surf_film.params['c'],
            c_world = self.surf_world.params['c'],
        )
        
        lens_mesh   = create_mesh(V_lens, F_lens, f"lens{self.id}")
        baffle_mesh = create_mesh(V_ap, F_ap, f"baffle{self.id}")
        film_mask   = mi.Mask(film_mask)
        world_mask  = mi.Mask(world_mask)

        lens_fname   = join(output_dir,   f'lens{self.id}.ply')
        baffle_fname = join(output_dir, f'baffle{self.id}.ply')
        lens_mesh  .write_ply(lens_fname)
        baffle_mesh.write_ply(baffle_fname)
        print('[+] Wrote lens mesh (subdivs={}) file to: {}'.format(self.subdiv_level, lens_fname))
        print('[+] Wrote baffle mesh file to: {}'.format(baffle_fname))
        
        self.film_vmask  = film_mask
        self.world_vmask = world_mask

        if (self.lens_fname is None) or (self.baffle_fname is None):
            self.lens_fname = lens_fname
            self.baffle_fname = baffle_fname


    def meshplot_geometry(self, p_ = None, lens_c = np.array([0,1,1]), baffle_c = np.array([0,1,0])) -> None:
        '''
        Visualize the lens geometry using Meshplot.
        '''
        if p_ is None:
            p_ = mp.plot(*read_triangle_mesh(self.lens_fname), c=lens_c)
        else:
            p_ .add_mesh(*read_triangle_mesh(self.lens_fname), c=lens_c)
            
        p_.add_mesh(*read_triangle_mesh(self.baffle_fname), c=baffle_c)

        return p_


    def add_to_scene(self, scene_dict) -> None:
        '''
        Register the lens element in the scene dictionary.
        '''
        lens_key = f"lens{self.id}"
        baffle_key = f"baffle{self.id}"

        lens_dict = {
                'type': 'ply',
                'id': lens_key,
                'filename': self.lens_fname,
                'bsdf': {'type': 'ref', 'id': 'simple-glass'},  # TODO: element tracks its own material
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
            self.param_keys_to_opt_keys = {'film': {}, 'world': {}}

        # iterate through *all* the shape params
        # surf_name = 'world' | 'film', 
        # param_dict = {'c': 1.0, 'K', 1.0, ...}
        # for surf_name, param_dict in self.shape_params.items():
        for surf_name, param_dict in zip(['film', 'world'], [self.surf_film.params, self.surf_world.params]):
            
            for var_name, value in param_dict.items():
                optvar_key = f'lens{self.id}_{surf_name}_{var_name}'
                if optvar_key in optimizer:
                    raise KeyError(f"Variable {optvar_key} already exists in optimizer!")
                else:
                    optimizer[optvar_key] = mi.Float(value)
                    self.param_keys_to_opt_keys[surf_name][var_name] = optvar_key
                    # TODO: add ability to select only some variables to optimize
        
                # TODO: handle per-variable learning rates


    def save_init_state(self, params: mi.SceneParameters):
        '''
        Must run this after the lens is added to the scene and the `scene` object is initialized.
        '''
        self.initial_vertex_pos = dr.unravel(mi.Point3f, params[f'{self.lens_key}.vertex_positions'])


    def update(self, params: mi.SceneParameters, optimizer: mi.ad.optimizers.Optimizer) -> None:
        '''
        Update lens element with the new values of the optimized variables, and recompute the lens geometry.
        '''
        # Compute new vertex positions for the `film` surface
        # copy existing shape params into a new dict
        # new_shape_params = self.shape_params['film'].copy()
        new_shape_params = self.surf_film.params

        # for params that are present in the optimizer, overwrite the old values with the optimizer's values
        for var_key, optvar_key in self.param_keys_to_opt_keys['film'].items():
            new_shape_params[var_key] = optimizer[optvar_key]

        new_vertex_pos = mi.Point3f(
                self.initial_vertex_pos[0], 
                self.initial_vertex_pos[1], 
                dr.select(self.film_vmask, 
                        self.surf_film.compute_z_dr(self.initial_vertex_pos[0], self.initial_vertex_pos[1]),
                        self.initial_vertex_pos[2]),
                        )

        # Handle the `world` surface in the same way
        # copy existing shape params into a new dict
        # new_shape_params = self.shape_params['world'].copy()
        new_shape_params = self.surf_world.params

        # for params that are present in the optimizer, overwrite the old values with the optimizer's values
        for var_key, optvar_key in self.param_keys_to_opt_keys['world'].items():
            new_shape_params[var_key] = optimizer[optvar_key]

        new_vertex_pos = mi.Point3f(
                new_vertex_pos[0], 
                new_vertex_pos[1], 
                dr.select(self.world_vmask, 
                        self.surf_world.compute_z_dr(new_vertex_pos[0], new_vertex_pos[1]),
                        new_vertex_pos[2]),
                        )

        # TODO: update baffle positions

        # Flatten the vertex position array before assigning it to `params`
        params[f'{self.lens_key}.vertex_positions'] = dr.ravel(new_vertex_pos)

        # Propagate changes through the scene (e.g. rebuild BVH)
        # NOTE: BVH update is performed in LensSystem


class LensSystem:
    def __init__(self, elements: tp.List[LensElement]):
        self.elements = elements
        self.rear_z = elements[0].surf_film.params['z0']
        self.front_z = elements[-1].surf_world.params['z0']

    def save_init_state(self, params: mi.SceneParameters):
        for element in self.elements:
            element.save_init_state(params)

    def update(self, params: mi.SceneParameters, optimizer: mi.ad.Optimizer) -> None:
        for element in self.elements:
            element.update(params, optimizer)

        # Update scene params at the end to rebuild the BVH
        params.update()

    def meshplot_geometry(self):
        '''
        Visualize the lens system's geometry using Meshplot.
        '''
        num_elements = len(self.elements)
        colors = np.c_[ np.ones(num_elements), 
                        np.ones(num_elements), 
                        np.arange(num_elements)]
        
        p_ = self.elements[0].meshplot_geometry(lens_c=colors[0], baffle_c=colors[0] * 0.5)

        for element in self.elements[1:]:
            c = colors[element.id]
            element.meshplot_geometry(p_=p_, lens_c=c, baffle_c=c * 0.5)

    def add_to_scene(self, scene_dict):
        for element in self.elements:
            element.add_to_scene(scene_dict)

    def add_to_optimizer(self, optimizer):
        for element in self.elements:
            element.add_to_optimizer(optimizer)

    def initialize_geometry(self, output_dir):
        for element in self.elements:
            element.initialize_geometry(output_dir)
