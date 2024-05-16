from __future__ import annotations
import sys
sys.path.insert(0, '/home/chuah/mitsuba3-camera/build/python')
sys.path.append("..")
import drjit as dr
import mitsuba as mi
import typing as tp
from scripts.lens_design.losses import rms_loss, color_loss
from scripts.lens_design.lens import LensSystem
import numpy as np

class FieldSource:
    def __init__(self, name, origin, target, radius, lens_rear_z, resolution, tracer):
        '''
        Field angle in degrees
        '''
        self.source_name = name
        self.sensor_name = "sensor_" + name
        self.origin = origin
        self.target = target
        self.radius = radius
        self.source_dict = self.get_source_dict(origin, target, radius)
        self.sensor_dict = self.get_sensor_dict(lens_rear_z, resolution, tracer)

    @staticmethod
    def createSourceArray(lens_system, resolution: tp.Tuple[int, int], tracer: tp.Callable, num_sources = 1, max_field_angle: float = 0.0):
        '''
        Max field angle in degrees
        '''
        source_radius = lens_system.front_radial_extent                  # TODO: EP radius
        entrance_pupil_position = np.array([0, 0, lens_system.entrance_pupil_position])  # TODO: EP z-position

        max_field_angle = np.deg2rad(np.abs(max_field_angle))
        field_angles = np.linspace(0.0, max_field_angle, num_sources)

        if num_sources > 1:
            dtheta = max_field_angle / (num_sources - 1)
            source_distance = 1.1 * source_radius / np.tan(0.5 * dtheta)
        else:
            source_distance = max(10.0, 5 * lens_system.front_z)

        sources = []
        for field_angle in field_angles:
            direction = np.array([np.sin(field_angle), 0.0, np.cos(field_angle)])
            origin = entrance_pupil_position + source_distance * direction
            source_name = f"field-{np.rad2deg(field_angle):04.1f}-deg".replace(".", "_")
            fs = FieldSource(
                name=source_name,
                origin=origin,
                target=entrance_pupil_position,
                radius=source_radius,
                lens_rear_z=lens_system.rear_z, 
                resolution=resolution, 
                tracer=tracer,
                )
            sources.append(fs)
            
        return sources

    def get_source_dict(self, origin, target, radius):
        # NOTE: replacing "directionalarea" with "directional" leads
        # to very poor sampling efficiency for ptracer (expected) while
        # providing no benefit for path tracing methods (path, prb)
        # emitter_dict = {
        #     'type':'directionalarea',
        #     'radiance': { 'type': 'spectrum', 'value': 0.05 },
        # }
        # sources['emitter-bsdf'] = emitter_dict

        source_dict = {
            'type': 'disk',

            # # enable mesh-based rep for scene visualization/debugging
            # 'type': 'obj',
            # 'filename': 'meshes/circle.obj',
            
            'to_world': mi.ScalarTransform4f.look_at(
                target=target,
                origin=origin,
                up=[0, 1, 0]) @ 
                mi.ScalarTransform4f.scale([radius, radius, 1]),
            'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
            'emitter': {
                'type':'directionalarea',
                'radiance': { 'type': 'spectrum', 'value': 0.05 },
            },
        }

        return source_dict
    
    def get_central_ray(self):
        direction = dr.normalize(mi.Point3f(self.target) - mi.Point3f(self.origin))
        return mi.Ray3f(o=self.origin, d=direction, wavelengths=[589.3])

    def get_sensor_dict(self, lens_rear_z, resolution, tracer):
        ray = self.get_central_ray()
        _, _, is_valid, film_pos = tracer(ray)
        film_pos = film_pos.numpy().ravel()
        is_valid = is_valid.numpy().ravel()

        if not(np.all(is_valid)):
            raise AssertionError(f"Oh no! {is_valid=}")
        
        near_clip = 0.1
        z_camera = max(0.3, 1.1 * near_clip)
        trafo_to_world = mi.ScalarTransform4f.look_at(
            target=[film_pos[0], film_pos[1], -1.0],
            # origin=[film_pos[0], film_pos[1], max(lens_rear_z - 1.0, 1.1 * near_clip)],
            origin=[film_pos[0], film_pos[1], z_camera],
            up=[0, 1, 0])
        
        sensor_dict = {
            # NOTE: orthographic doesn't work bc emitted importance is not implemented
            'type': 'perspective',
            'near_clip': near_clip,
            'far_clip': 10,
            'fov': 45,
            'to_world': trafo_to_world,

            'sampler': {
                'type': 'independent',
                'sample_count': 512  # Not really used
            },
            'film': {
                'type': 'hdrfilm',
                'width':  resolution[0],
                'height': resolution[1],
                'pixel_format': 'rgb',
                'rfilter': {
                    'type': 'tent'
                    }
                },
            }
        
        init_pixel_size = z_camera * dr.deg2rad(0.5 * sensor_dict['fov']) / resolution[0]
        self.init_pixel_area = init_pixel_size ** 2
        
        return sensor_dict
    
    # def get_pixel_size(self):
    #     z_camera = self.origin[2]
    #     resolution = self.sensor_dict['film']['width']
    #     pixel_size = z_camera * dr.deg2rad(0.5 * self.sensor_dict['fov']) / resolution
    #     return pixel_size
    


class DesignProblem:
    def __init__(self, 
                 lens_system: LensSystem,
                 resolution: tp.Tuple[int, int], 
                 spp: int, 
                 learning_rate: float, 
                 iters: int,
                 film_diagonal: int = 35,
                 ):
        self.resolution = resolution
        self.spp = spp
        self.lr = learning_rate
        self.iters = iters
        self.film_diagonal = film_diagonal
        self.lens_system = lens_system

        # TODO  XXXXX
        self.num_sources = 5
        self.max_field_angle = 5.0

    def add_field_sources(self, scene, geo_tracer):
        '''
        Add endpoints for all the field sources in the scene:
        - directional emitters
        - camera sensors
        '''
        field_sources = FieldSource.createSourceArray(
            lens_system=self.lens_system, 
            num_sources=self.num_sources, 
            max_field_angle=self.max_field_angle,
            resolution=self.resolution,
            tracer=geo_tracer)

        for source in field_sources:
            scene[source.source_name] = source.source_dict
            scene[source.sensor_name] = source.sensor_dict

        self.field_sources = field_sources

    def upsample_sensor(self, idx, zoom_factor = 2.0):
        '''Updates the sensor of index `idx` to do a 2x zoom on the center of the image field.
        '''
        fov_key = f'{self.sensor_names[idx]}.x_fov'
        curr_fov = self.params[fov_key]
        zoom_fov = dr.rad2deg(2 * dr.atan(dr.rcp(zoom_factor) * dr.tan(0.5 * dr.deg2rad(curr_fov))))
        self.params[fov_key] = zoom_fov

    def upsample_sensors(self, zoom_factor = 2.0):
        for idx in range(self.num_sources):
            self.upsample_sensor(idx, zoom_factor)


    def _get_integrator(self, integrator_type: str):
        max_depth = self.lens_system.size() + 2
        integrator = {
            'type': integrator_type,
            'max_depth': max_depth,
            'hide_emitters': False,
        }
        return integrator
    
    def _build_preliminary_scene(self, integrator_type: str):
        '''
        Build a preliminary version of the scene with lenses and film plane only.
        '''
        integrator = self._get_integrator(integrator_type)
        preliminary_scene = {
            'type': 'scene',
            'integrator': integrator,
            'white-bsdf': {
                'type': 'diffuse',
                'id': 'white-bsdf',
                'reflectance': { 'type': 'rgb', 'value': (1, 1, 1) },
            },
            'black-bsdf': {
                'type': 'diffuse',
                'id': 'black-bsdf',
                'reflectance': { 'type': 'spectrum', 'value': 0 },
            },
            # 'emitter-bsdf': {
            #     'type':'directionalarea',
            #     'radiance': { 'type': 'spectrum', 'value': 0.05 },
            #     'id': 'emitter-bsdf',
            # },
            # Film plane
            'film_plane': {
                'type': 'obj',
                'id': 'film_plane',
                'filename': 'meshes/rectangle.obj',
                'to_world': \
                    mi.ScalarTransform4f.look_at(
                        target=[0, 0, 1],
                        origin=[0, 0, 0],
                        up=[0, 1, 0]
                    ).scale((self.film_diagonal, self.film_diagonal, 1)),
                'bsdf': {'type': 'ref', 'id': 'white-bsdf'},
            },
        }
        self.lens_system.add_to_scene(preliminary_scene)

        return preliminary_scene
    
    def _build_scene(self):
        '''
        Integrator type: prb_basic for ray paths, ptracer for rendering
        '''
        # version of the scene for geometric raytracing; a separate scene instance is
        # needed because the "trace" method is only implemented in `prb_basic`, not 
        # `ptracer` (which we use for regular rendering)
        geo_scene = mi.load_dict(self._build_preliminary_scene("prb_basic"))
        geo_tracer = lambda ray: geo_scene.integrator().trace(geo_scene, ray, self.lens_system.size() + 1)
        # self.tracer = lambda ray, N: geo_scene.integrator().trace(geo_scene, ray, N)

        # version of the scene used for rendering
        scene = self._build_preliminary_scene("ptracer")
        self.add_field_sources(scene, geo_tracer)
        scene = mi.load_dict(scene)
        self.scene = scene

        self.__test_geo_tracer(geo_tracer)

        del geo_tracer, geo_scene
        return scene
    
    def __test_geo_tracer(self, tracer):
        ls = self.lens_system

        z_init = ls.front_z * 1.1
        r_max = ls.front_radial_extent * 0.001
        theta_max = 0.001 * np.pi

        N = 100

        rs = r_max * (2.0 * mi.radical_inverse_2(dr.arange(mi.UInt32, N), mi.UInt32(0)) - 1.0)
        thetas = theta_max * (2.0 * mi.sobol_2(dr.arange(mi.UInt32, N), mi.UInt32(0)) - 1.0)

        # negate direction since ray is traveling backwards
        dir = -mi.Vector3f(dr.tan(thetas), 0.0, 1.0)
        ray = mi.Ray3f(o = mi.Point3f(rs, 0.0, z_init),
                       d = dr.normalize(dir),
                       wavelengths = [589.3])
        
        _, d_exact, _, out_exact = tracer(ray)
        
        # paraxial setup
        y = rs - (z_init - ls.front_z) * np.tan(thetas)
        u = np.tan(thetas)
        ray_paraxial = mi.Vector2f(y, u)
        out_paraxial = ls.trace_paraxial_front2film(ray_paraxial)

        # print(out_exact.x)
        # print(out_paraxial.x)
        print(dr.max(dr.abs(out_exact.x - out_paraxial.x)))
        print("y is close: ", dr.allclose(out_exact.x, out_paraxial.x, atol=2e-5, rtol=1e-8))
        # print(d_exact.x / d_exact.z)
        # print(out_paraxial.y)
        print(dr.max(dr.abs(d_exact.x / d_exact.z - out_paraxial.y)))
        print("u is close: ", dr.allclose(d_exact.x / d_exact.z, out_paraxial.y, atol=2e-5, rtol=1e-8))

        return out_exact, out_paraxial



    def reset(self):
        self.scene = None
        self.params = None
        self.optimizer = None

    def prepare(self):
        with dr.suspend_grad():
            self.reset()
            self.scene = self._build_scene()
            images = [mi.render(self.scene, 
                                spp=max(self.spp, 512), 
                                sensor=sensor_idx) 
                        for sensor_idx in range(self.num_sources)]
            plt.figure(figsize=(12,6))
            for sensor_idx, image in enumerate(images):
                plt.subplot(1, self.num_sources, sensor_idx + 1)
                # mi.util.write_bitmap("test_image.exr", test_image)
                plt.imshow(image)
                emitter_name = self.scene.emitters()[sensor_idx].shape().id().replace("_",".")
                plt.title(f"sensors[{sensor_idx}]: {emitter_name}")

        params = mi.traverse(self.scene)
        optimizer = mi.ad.Adam(lr=self.lr)
        self.lens_system.save_init_state(params)
        self.lens_system.add_to_optimizer(optimizer)
        self.params = params
        self.optimizer = optimizer
        self.sensor_names = [self.scene.sensors()[sensor_idx].id() for sensor_idx in range(self.num_sources)]


    def optimize(self):
        mi.set_log_level(mi.LogLevel.Warn)
        loss_values = []
        spp = self.spp
        upsample_steps = 0

        with dr.suspend_grad():
            images_init = [mi.render(self.scene, self.params, sensor=i) for i in range(self.num_sources)]

        for it in range(self.iters):
            # Apply displacement and update the scene BHV accordingly
            self.lens_system.update(self.params, self.optimizer)

            loss = mi.Float(0.0)
            upsample_flag = True
            for sensor_idx in range(self.num_sources):
                # Perform a differentiable rendering of the scene
                image = mi.render(self.scene, 
                                  self.params, 
                                  seed=it, 
                                  spp=2 * spp, 
                                  spp_grad=spp,
                                  sensor=sensor_idx)

                # Scale-independent L2 function
                sensor_rms_loss = rms_loss(image)
                upsample_flag &= sensor_rms_loss[0] < dr.sqr(self.resolution[0] * 0.1)
                sensor_color_loss = color_loss(image)

                # XXXXX
                sensor_rms_loss *= self.field_sources[sensor_idx].init_pixel_area
                print(sensor_rms_loss, sensor_color_loss)
                loss += sensor_rms_loss + sensor_color_loss

            # Back-propagate errors to input parameters and take an optimizer step
            dr.backward(loss)

            # Take a gradient step
            self.optimizer.step()

            # Log progress
            current_loss = loss[0] / (4 ** upsample_steps)
            loss_values.append(current_loss)
            mi.Thread.thread().logger().log_progress(
                it / (self.iters-1),
                f'Iteration {it:03d}: loss={current_loss:g}', 'Caustic Optimization', '')

            if upsample_flag:
                upsample_steps += 1
                self.upsample_sensors()
                self.params.update(self.optimizer)
                print(f"Iter {it}: upsampling to level {upsample_steps} due to {loss[0] = }")
                print(self.params[f'{self.sensor_names[0]}.x_fov'])
            else:
                self.params.update(self.optimizer)

            # Increase rendering quality toward the end of the optimization
            if it in (int(0.7 * self.iters), int(0.9 * self.iters)):
                # spp *= 2
                self.optimizer.set_learning_rate(0.5 * self.optimizer.lr_default)


        with dr.suspend_grad():
            images_final = [mi.render(self.scene, self.params, spp=2 * spp, sensor=i) for i in range(self.num_sources)]
    
        return loss_values, images_final, images_init, upsample_steps

    def render(self, spp, sensor_idx = 0, resolution = None, zoom_factor = None):
        with dr.suspend_grad():
            if zoom_factor is not None:
                self.upsample_sensors(zoom_factor)

            if resolution is not None:
                var_key = f"{self.sensor_names[sensor_idx]}.film.size"
                curr_resolution = (self.params[var_key][0], self.params[var_key][1])
                crop_size = self.params[f"{self.sensor_names[sensor_idx]}.film.crop_size"]
                crop_offset = self.params[f"{self.sensor_names[sensor_idx]}.film.crop_offset"]
                self.params[var_key] = (resolution[0], resolution[1])
                self.params.update(self.optimizer)

            image = mi.render(self.scene, 
                        self.params, 
                        spp=spp, 
                        sensor=sensor_idx)

            if zoom_factor is not None:
                self.upsample_sensors(1 / zoom_factor)

            if resolution is not None:
                self.params[var_key] = curr_resolution
                self.params[f"{self.sensor_names[sensor_idx]}.film.crop_size"] = crop_size
                self.params[f"{self.sensor_names[sensor_idx]}.film.crop_offset"] = crop_offset
                self.params.update(self.optimizer)

        return image



from typing_extensions import override

class ConstrainedEFLProblem(DesignProblem):
    def __init__(self, 
                 lens_system: LensSystem,
                 resolution: tp.Tuple[int, int], 
                 spp: int, 
                 learning_rate: float, 
                 target_focal_length: float,
                 iters: int,
                 output_dir: str,
                 film_diagonal: int = 35,
                 ):
        
        self.target_focal_length = target_focal_length

        # disable optimization of the rear surface's 1st order parameters
        lens_system.disable_var_in_surface('c', 0)
        # TODO: could still let z0 be free to get a better focus than the paraxial focus
        lens_system.disable_var_in_surface('z0', 0)

        # set first surface
        s1_params = lens_system.get_rear_surface_params(target_focal_length)
        lens_system.elements[0].surface.params['c']  = s1_params['c' ].numpy().item()
        lens_system.elements[0].surface.params['z0'] = s1_params['z0'].numpy().item()
        lens_system.compute_paraxial_quantities()

        print("Focal length target: ", 
                lens_system.compute_EFL(), " vs. ",
                target_focal_length)
        print("BFL focus target: ", 
                lens_system.compute_BFL(), " vs. ",
                lens_system.elements[0].surface.params['z0'])
        
        lens_system.initialize_geometry(output_dir)
        lens_system.disable_all_materials()

        super().__init__(lens_system, resolution, spp, learning_rate, iters, film_diagonal)

    @override
    def optimize(self):
        # print(self.optimizer.variables)
        mi.set_log_level(mi.LogLevel.Warn)
        loss_values = []
        spp = self.spp
        upsample_steps = 0
        with dr.suspend_grad():
            images_init = [mi.render(self.scene, self.params, sensor=i) for i in range(self.num_sources)]

        for it in range(self.iters):
            # Apply displacement and update the scene BHV accordingly
            self.lens_system.fixed_EFL_update(self.params, self.optimizer, self.target_focal_length)
            # print(self.optimizer.variables)

            loss = mi.Float(0.0)
            upsample_flag = True
            for sensor_idx in range(self.num_sources):
                # Perform a differentiable rendering of the scene
                image = mi.render(self.scene, 
                                  self.params, 
                                  seed=it, 
                                  spp=2 * spp, 
                                  spp_grad=spp,
                                  sensor=sensor_idx)

                # RMS loss function
                sensor_loss = rms_loss(image)
                # fix for nans: leave out the higher order aspheres
                # print("loss nan?: ", sensor_loss)
                # print("isnan: ", np.any(np.isnan(image.numpy())))
                upsample_flag &= sensor_loss[0] < dr.sqr(self.resolution[0] * 0.1)
                
                sensor_loss *= dr.detach(self.field_sources[sensor_idx].init_pixel_area / (4 ** upsample_steps))
                loss += 10000 * sensor_loss


                # XXXXX
                # the color loss appears to be bad for rms!!!
                
                # # RMS and dispersion loss function
                # sensor_rms_loss = rms_loss(image)
                # upsample_flag &= sensor_rms_loss[0] < dr.sqr(self.resolution[0] * 0.1)
                # sensor_rms_loss *= dr.detach(self.field_sources[sensor_idx].init_pixel_area / (4 ** upsample_steps))

                # # # print(sensor_rms_loss / self.resolution[0], sensor_color_loss)
                # # loss += sensor_rms_loss / self.resolution[0] + 1000 * sensor_color_loss

                # sensor_color_loss = color_loss(image)
                # loss += 1e4 * sensor_rms_loss + sensor_color_loss
                # # print(sensor_rms_loss, sensor_color_loss)


            # Back-propagate errors to input parameters and take an optimizer step
            dr.backward(loss)

            # Take a gradient step
            self.optimizer.step()

            # Log progress
            current_loss = loss[0] # / (4 ** upsample_steps)
            loss_values.append(current_loss)
            mi.Thread.thread().logger().log_progress(
                it / (self.iters-1),
                f'Iteration {it:03d}: loss={current_loss:g}', 'Caustic Optimization', '')

            if upsample_flag:
                upsample_steps += 1
                self.upsample_sensors()
                self.params.update(self.optimizer)
                print(f"Iter {it}: upsampling to level {upsample_steps} due to {loss[0] = }")
                print(self.params[f'{self.sensor_names[0]}.x_fov'])
            else:
                self.params.update(self.optimizer)

            # Increase rendering quality toward the end of the optimization
            if it in (int(0.5 * self.iters), int(0.7 * self.iters)):
                # spp *= 2
                self.optimizer.set_learning_rate(0.1 * self.optimizer.lr_default)

        with dr.suspend_grad():
            images_final = [mi.render(self.scene, self.params, spp=2 * spp, sensor=i) for i in range(self.num_sources)]
    
        return loss_values, images_final, images_init, upsample_steps



import matplotlib.pyplot as plt
import numpy as np

def show_image(ax, img, title):
    ax.imshow(mi.util.convert_to_bitmap(img))
    ax.axis('off')
    ax.set_title(title)

def plot_progress(image_init, image_final, loss_values, upsample_steps, average_spot=True):
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    ax = ax.ravel()
    ax[0].semilogy(loss_values)
    ax[0].set_xlabel('Iteration'); ax[0].set_ylabel('Loss value'); ax[0].set_title('Convergence plot')
    spot_size_init = np.sqrt(rms_loss(image_init).numpy()).item() * (2 ** upsample_steps)
    spot_size_final = np.sqrt(rms_loss(image_final).numpy()).item()
    show_image(ax[2], image_init, f'Initial spot size: {spot_size_init:.3f}')
    show_image(ax[3], image_final / (30 * np.mean(image_final)),      f'Final spot size: {spot_size_final:.3f}')
    # show_image(ax[3], image_final,      f'Final spot size: {np.sqrt(rms_loss(image_final).numpy().item()):.3f}')



    spot_final = image_final.numpy()[image_final.shape[0] // 2]

    image_init_np = np.array(image_init)
    resolution = (image_init.shape[0] * 2 ** upsample_steps, image_init.shape[1] * 2 ** upsample_steps)
    mi.set_variant('scalar_rgb')
    rfilter = mi.scalar_rgb.load_dict({'type': 'box'})
    tmp = np.array(mi.Bitmap(image_init_np).resample(resolution, rfilter))
    spot_init = tmp[resolution[0] // 2]

    if average_spot:
        spot_init  = np.mean(spot_init,  axis=1)
        spot_final = np.mean(spot_final, axis=1)

    res_x_init = spot_init.shape[0]
    res_x_final = spot_final.shape[0]

    spot_init /= np.max(spot_init)
    spot_final /= np.max(spot_final)

    ax[1].plot(np.arange(res_x_init), spot_init, label="Init")
    ax[1].plot(np.arange(res_x_final) - 0.5 * res_x_final + 0.5 * res_x_init, spot_final, label="Final")
    ax[1].set_title('Spot profile')
    ax[1].set_ylabel("Norm. radiance (a.u.)")
    ax[1].legend(loc='upper right')

    mi.set_variant('cuda_ad_dispersion')

    return fig

from scripts.lens_design.geometry import meshplot_gizmo
from gpytoolbox import cone
import meshplot as mp

def draw_camera(sensor_to_world):
    # plot sensor as a +z-facing cone
    V, F = cone(16,2)
    V += np.array([[0,0,-1]])
    V *= -1

    # apply camera-to-world transform
    V = (sensor_to_world @ mi.Point3f(V)).numpy()
    p_ = mp.plot(V,F)
    meshplot_gizmo(p_)
    return p_

def meshplot_scene(problem):
    params = problem.params
    lens_system = problem.lens_system
    sources = problem.field_sources

    p_ = draw_camera(params['sensor.to_world'])
    meshes = ['camera_housing_0', 
              'camera_housing_1', 
              'camera_housing_2', 
              'camera_housing_3', 
              'film_plane']
    
    for source in sources:
        meshes += [source.source_name]

    for element in lens_system.elements:
        meshes += [element.lens_key, element.baffle_key]
    for mesh in meshes:
        V = dr.unravel(mi.Point3f, params[f'{mesh}.vertex_positions']).numpy()
        F = dr.unravel(mi.Point3u, params[f'{mesh}.faces']).numpy()
        p_.add_mesh(V, F, shading={"wireframe": True, "side": "FrontSide"})
    return p_

