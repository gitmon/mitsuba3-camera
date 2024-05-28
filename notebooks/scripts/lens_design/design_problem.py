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
    def __init__(self, 
                 name: str, 
                 origin: np.ndarray, 
                 target: np.ndarray, 
                 radius: np.ndarray, 
                 lens_system: LensSystem, 
                 resolution: tuple[int, int], 
                 # TODO
                 tracer,
                 ):
        '''
        Field angle in degrees
        '''
        self.source_key = "source_" + name
        self.sensor_key = "sensor_" + name
        # shared data
        self.origin = origin
        self.target = target
        # source
        self.radius = radius
        # sensor
        self.near_clip = 0.1
        self.resolution = resolution
        self.fov = 45.0
        self.camera_pos = [0,0,0]

        self.lens_system = lens_system

        # TODO
        self.sensor_dict = self.get_sensor_dict(tracer)
        self.losses = []

    def add_to_scene(self, scene_dict: dict) -> None:
        '''
        Register the field source in the scene dictionary.
        '''
        source_key = self.source_key
        sensor_key = self.sensor_key

        source_dict = {
            'type': 'disk',
            'to_world': mi.ScalarTransform4f.look_at(
                target=self.target,
                origin=self.origin,
                up=[0, 1, 0]) @ 
                mi.ScalarTransform4f.scale([self.radius, self.radius, 1]),
            'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
            'emitter': {
                'type':'directionalarea',
                'radiance': { 'type': 'spectrum', 'value': 0.05 },
            },
        }

        if source_key not in scene_dict:
            scene_dict[source_key] = source_dict
        else:
            raise KeyError(f"Source `{source_key}` already exists in scene!")

        # sensor_dict = self.get_sensor_dict(lens_rear_z, tracer)
        # sensor_dict = self.sensor_dict

        if sensor_key not in scene_dict:
            scene_dict[sensor_key] = self.sensor_dict
        else:
            raise KeyError(f"Sensor `{sensor_key}` already exists in scene!")


    def update(self, params: mi.SceneParameters):
        '''
        Update the source and sensor data when the lens system undergoes a gradient step.
        NOTE: currently, the source and sensor positions are not updated when the lens geometry
        changes. To be fully correct, we should:
        - Update pupil position -> update source's origin and target -> update central ray ->
            trace central ray -> update sensor position
        
        Ideally, this should be done at every iteration.
        '''
        # # source update
        # self.target = [0, 0, self.lens_system.entrance_pupil_position]
        # self.origin = self.target + self.direction * d
        # source_to_world = mi.ScalarTransform4f.look_at(
        #     target=self.target,
        #     origin=self.origin,
        #     up=[0, 1, 0]) @ mi.ScalarTransform4f.scale([self.radius, self.radius, 1])     

        # # sensor update
        # ray = self.get_central_ray()
        # _, _, is_valid, film_pos = tracer(ray)
        # sensor_pos = [film_pos[0], film_pos[1], max(0.8 * lens_rear_z, 1.1 * self.near_clip)]
        # sensor_to_world = mi.ScalarTransform4f.look_at(
        #     target=[film_pos[0], film_pos[1], -1.0],
        #     origin=sensor_pos,
        #     up=[0, 1, 0])

        with dr.suspend_grad():
            # # source update
            # params[f"{self.source_key}.to_world"] = source_to_world

            # # sensor update
            # update transform
            # params[f"{self.sensor_key}.to_world"] = sensor_to_world
            # update FOV for upsampling
            params[f"{self.sensor_key}.x_fov"] = self.fov


    def upsample_sensor(self, zoom_factor = 2.0):
        '''
        Updates the sensor to perform a 2x zoom on the center of the image field.
        '''
        self.fov = dr.rad2deg(2 * dr.atan(dr.rcp(zoom_factor) * dr.tan(0.5 * dr.deg2rad(self.fov))))

    @staticmethod
    def createSourceArray(lens_system, resolution: tp.Tuple[int, int], tracer: tp.Callable, num_sources = 1, max_field_angle: float = 0.0):
        '''
        Max field angle in degrees
        '''
        source_radius = lens_system.front_radial_extent                  # TODO: EP radius?
        entrance_pupil_position = np.array([0, 0, lens_system.entrance_pupil_position])

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
            source_name = f"{np.rad2deg(field_angle):04.1f}-deg".replace(".", "_")
            fs = FieldSource(
                name=source_name,
                origin=origin,
                target=entrance_pupil_position,
                radius=source_radius,
                lens_system=lens_system,
                resolution=resolution,
                tracer=tracer,
                )
            sources.append(fs)
            
        return sources

    def get_central_ray(self):
        direction = dr.normalize(mi.Point3f(self.target) - mi.Point3f(self.origin))
        return mi.Ray3f(o=self.origin, d=direction, wavelengths=[589.3])

    def get_sensor_dict(self, tracer):
        '''
        Get the sensor's scene data.

        TODO: Note that the sensor's position (both z/axial and lateral) depends on the 
        path of the source's central ray, and is thus a function of the lens system geometry.

        '''
        ray = self.get_central_ray()
        _, _, is_valid, film_pos = tracer(ray)
        film_pos = film_pos.numpy().ravel()
        is_valid = is_valid.numpy().ravel()

        if not(np.all(is_valid)):
            raise AssertionError(f"Source's central ray was not transmitted through the lens system! {is_valid=}")
        
        z_camera = max(0.02 * self.lens_system.rear_z, 1.1 * self.near_clip)
        self.camera_pos = [film_pos[0], film_pos[1], z_camera]

        trafo_to_world = mi.ScalarTransform4f.look_at(
            target=[film_pos[0], film_pos[1], -1.0],
            origin=self.camera_pos,
            up=[0, 1, 0])
        
        sensor_dict = {
            'type': 'perspective',
            'near_clip': self.near_clip,
            'far_clip': z_camera + 1.0,
            'fov': self.fov,
            'to_world': trafo_to_world,

            'sampler': {
                'type': 'independent',
                'sample_count': 512  # Not really used. TODO: remove?
            },
            'film': {
                'type': 'hdrfilm',
                'width':  self.resolution[0],
                'height': self.resolution[1],
                'pixel_format': 'rgb',
                'rfilter': {
                    'type': 'tent'
                    }
                },
            }
        
        self.init_pixel_size = self.camera_pos[2] * dr.tan(dr.deg2rad(0.5 * self.fov)) / self.resolution[0]
        
        return sensor_dict
    
    def get_pixel_size(self):
        '''
        Get the physical size of a camera pixel in millimeters.
        '''
        z_camera = self.camera_pos[2]
        pixel_size = z_camera * dr.tan(dr.deg2rad(0.5 * self.fov)) / self.resolution[0]
        return pixel_size
    
    def evaluate_rms_loss(self, image):
        '''
        Compute the physical RMS spot size as well as the upsample flag.
        '''
        pixel_loss = rms_loss(image)
        upsample_flag = pixel_loss[0] < dr.sqr(self.resolution[0] * 0.1)
        physical_loss = pixel_loss * self.get_pixel_size() ** 2
        self.losses.append(physical_loss)
        return physical_loss, upsample_flag
    


class DesignProblem:
    def __init__(self, 
                 lens_system: LensSystem,
                 resolution: tp.Tuple[int, int], 
                 spp: int, 
                 learning_rate: float, 
                 iters: int,
                 output_dir: str,
                 film_diagonal: int = 35,
                 ):

        lens_system.initialize_geometry(output_dir)

        self.resolution = resolution
        self.spp = spp
        self.lr = learning_rate
        self.iters = iters
        self.film_diagonal = film_diagonal
        self.lens_system = lens_system

        # TODO  XXXXX
        self.num_sources = 1
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

        for field_source in field_sources:
            field_source.add_to_scene(scene)

        self.field_sources = field_sources

        # field_sources = FieldSource.createSourceArray(
        #     lens_system=self.lens_system, 
        #     num_sources=self.num_sources, 
        #     max_field_angle=self.max_field_angle,
        #     resolution=self.resolution,
        #     tracer=geo_tracer)

        # for source in field_sources:
        #     scene[source.source_name] = source.source_dict
        #     scene[source.sensor_name] = source.sensor_dict

        # self.field_sources = field_sources

    # def upsample_sensor(self, idx, zoom_factor = 2.0):
    #     '''Updates the sensor of index `idx` to do a 2x zoom on the center of the image field.
    #     '''
    #     fov_key = f'{self.sensor_names[idx]}.x_fov'
    #     curr_fov = self.params[fov_key]
    #     zoom_fov = dr.rad2deg(2 * dr.atan(dr.rcp(zoom_factor) * dr.tan(0.5 * dr.deg2rad(curr_fov))))
    #     self.params[fov_key] = zoom_fov

    # def upsample_sensors(self, zoom_factor = 2.0):
    #     for idx in range(self.num_sources):
    #         self.upsample_sensor(idx, zoom_factor)


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

        Returns:
        - preliminary_scene: dict. Preliminary scene dict.
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

        Returns:
        - scene: mi.Scene. The initialized scene.
        '''
        # version of the scene for geometric raytracing; a separate scene instance is
        # needed because the "trace" method is only implemented in `prb_basic`, not 
        # `ptracer` (which we use for regular rendering)
        # TODO: ideally, the tracer should come packaged/be a method of `lens_system` and
        #       use exact raytracing of the analytic surfaces
        geo_scene = mi.load_dict(self._build_preliminary_scene("prb_basic"))
        geo_tracer = lambda ray: geo_scene.integrator().trace(geo_scene, ray, self.lens_system.size() + 1)

        # version of the scene used for rendering
        scene = self._build_preliminary_scene("ptracer")
        self.add_field_sources(scene, geo_tracer)
        scene = mi.load_dict(scene)
        self.scene = scene

        self.__test_geo_tracer(geo_tracer)

        del geo_tracer, geo_scene
        return scene
    
    def __test_geo_tracer(self, tracer):
        '''
        Test the scene's numeric raytracing against paraxial raytracing for the lens system.
        '''
        lens_sys = self.lens_system

        z_init = lens_sys.front_z * 1.1
        r_max = lens_sys.front_radial_extent * 0.001
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
        y = rs - (z_init - lens_sys.front_z) * np.tan(thetas)
        u = np.tan(thetas)
        ray_paraxial = mi.Vector2f(y, u)
        out_paraxial = lens_sys.trace_paraxial_front2film(ray_paraxial)

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


    def optimize(self, save_var_history=False):
        mi.set_log_level(mi.LogLevel.Warn)
        loss_values = []
        spp = self.spp
        upsample_steps = 0

        if save_var_history:
            vars = []

        with dr.suspend_grad():
            images_init = [mi.render(self.scene, self.params, sensor=i) for i in range(self.num_sources)]
            spots_init = [dr.sqrt(self.field_sources[i].evaluate_rms_loss(img)[0])[0] for i, img in enumerate(images_init)]

        for it in range(self.iters):
            # Apply displacement and update the scene BHV accordingly
            self.lens_system.update(self.params, self.optimizer)

            # Update the field sources (upsampling, new spot positions, etc.)
            for fs in self.field_sources:
                fs.update(self.params)

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

                # # Scale-independent L2 function
                # sensor_rms_loss, upsample = self.field_sources[sensor_idx].evaluate_rms_loss(image)
                # upsample_flag &= upsample
                # # sensor_rms_loss = rms_loss(image)
                # # upsample_flag &= sensor_rms_loss[0] < dr.sqr(self.resolution[0] * 0.1)
                # sensor_color_loss = color_loss(image)

                # # XXXXX
                # # sensor_rms_loss *= self.field_sources[sensor_idx].init_pixel_area
                # # print(sensor_rms_loss, sensor_color_loss)
                # loss += sensor_rms_loss + sensor_color_loss

                sensor_loss, upsample = self.field_sources[sensor_idx].evaluate_rms_loss(image)
                upsample_flag &= upsample
                loss += 10000 * sensor_loss


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
                for fs in self.field_sources:
                    fs.upsample_sensor()
                self.params.update(self.optimizer)
                print(f"Iter {it}: upsampling to level {upsample_steps} due to {loss[0] = }")
            else:
                self.params.update(self.optimizer)

            # Increase rendering quality toward the end of the optimization
            if it in (int(0.7 * self.iters), int(0.9 * self.iters)):
                # spp *= 2
                self.optimizer.set_learning_rate(0.3 * self.optimizer.lr_default)

            if save_var_history:
                vars.append([dr.detach(x).numpy() for x in self.optimizer.variables.values()])



        with dr.suspend_grad():
            images_final = [mi.render(self.scene, self.params, spp=2 * spp, sensor=i) for i in range(self.num_sources)]
            spots_final = [dr.sqrt(self.field_sources[i].evaluate_rms_loss(img)[0])[0] for i, img in enumerate(images_final)]

        if save_var_history:
            return loss_values, images_final, images_init, upsample_steps, spots_init, spots_final, vars
        else:
            return loss_values, images_final, images_init, upsample_steps, spots_init, spots_final

    def render(self, spp, sensor_idx = 0, resolution = None, zoom_factor = None):
        with dr.suspend_grad():
            if zoom_factor is not None:
                for fs in self.field_sources:
                    fs.upsample_sensor(zoom_factor)
                    fs.update(self.params)
                # self.upsample_sensors(zoom_factor)

            if resolution is not None:
                film_key = f"{self.field_sources[sensor_idx].sensor_key}.film"
                res_key = f"{film_key}.size"
                curr_resolution = (self.params[res_key][0], self.params[res_key][1])
                crop_size = self.params[f"{film_key}.crop_size"]
                crop_offset = self.params[f"{film_key}.crop_offset"]
                self.params[res_key] = (resolution[0], resolution[1])
                self.params.update(self.optimizer)

            image = mi.render(self.scene, 
                        self.params, 
                        spp=spp, 
                        sensor=sensor_idx)

            if zoom_factor is not None:
                for fs in self.field_sources:
                    fs.upsample_sensor(1 / zoom_factor)
                    fs.update(self.params)
                # self.upsample_sensors(1 / zoom_factor)

            if resolution is not None:
                self.params[res_key] = curr_resolution
                self.params[f"{film_key}.crop_size"] = crop_size
                self.params[f"{film_key}.crop_offset"] = crop_offset
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
        
        # lens_system.initialize_geometry(output_dir)
        # lens_system.disable_all_materials()

        super().__init__(lens_system, resolution, spp, learning_rate, iters, output_dir, film_diagonal)

    @override
    def optimize(self, save_var_history=False):
        # print(self.optimizer.variables)
        mi.set_log_level(mi.LogLevel.Warn)
        loss_values = []
        spp = self.spp
        upsample_steps = 0

        if save_var_history:
            vars = []

        with dr.suspend_grad():
            images_init = [mi.render(self.scene, self.params, sensor=i) for i in range(self.num_sources)]
            spots_init = [dr.sqrt(self.field_sources[i].evaluate_rms_loss(img)[0])[0] for i, img in enumerate(images_init)]

        for it in range(self.iters):
            # Apply displacement and update the scene BHV accordingly
            self.lens_system.fixed_EFL_update(self.params, self.optimizer, self.target_focal_length)
            # print(self.optimizer.variables)

            # Update the field sources (upsampling, new spot positions, etc.)
            for fs in self.field_sources:
                fs.update(self.params)

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
                # sensor_loss = rms_loss(image)
                # upsample_flag &= sensor_loss[0] < dr.sqr(self.resolution[0] * 0.1)
                # sensor_loss *= dr.detach(self.field_sources[sensor_idx].init_pixel_area / (4 ** upsample_steps))
                sensor_loss, upsample = self.field_sources[sensor_idx].evaluate_rms_loss(image)
                upsample_flag &= upsample
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
                for fs in self.field_sources:
                    fs.upsample_sensor()
                self.params.update(self.optimizer)
                print(f"Iter {it}: upsampling to level {upsample_steps}")
            else:
                self.params.update(self.optimizer)

            # Increase rendering quality toward the end of the optimization
            if it in (int(0.5 * self.iters), int(0.7 * self.iters)):
                # spp *= 2
                self.optimizer.set_learning_rate(0.1 * self.optimizer.lr_default)

            if save_var_history:
                vars.append([dr.detach(x).numpy() for x in self.optimizer.variables.values()])

        with dr.suspend_grad():
            images_final = [mi.render(self.scene, self.params, spp=2 * spp, sensor=i) for i in range(self.num_sources)]
            spots_final = [dr.sqrt(self.field_sources[i].evaluate_rms_loss(img)[0])[0] for i, img in enumerate(images_final)]
    
        if save_var_history:
            return loss_values, images_final, images_init, upsample_steps, spots_init, spots_final, vars
        else:
            return loss_values, images_final, images_init, upsample_steps, spots_init, spots_final


import matplotlib.pyplot as plt
import numpy as np

def show_image(ax, img, title):
    ax.imshow(mi.util.convert_to_bitmap(img))
    ax.axis('off')
    ax.set_title(title)

def plot_progress(image_init, image_final, spot_size_init, spot_size_final, loss_values, upsample_steps, average_spot=True):
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    ax = ax.ravel()
    ax[0].semilogy(loss_values)
    ax[0].set_xlabel('Iteration'); ax[0].set_ylabel('Loss value'); ax[0].set_title('Convergence plot')
    # show_image(ax[2], image_init, f'Initial spot size: {spot_size_init:.3f}')
    # show_image(ax[3], image_final / (30 * np.mean(image_final)),      f'Final spot size: {spot_size_final:.3f}')


    image_init_np = np.array(image_init)
    resolution = (image_init.shape[0] * 2 ** upsample_steps, image_init.shape[1] * 2 ** upsample_steps)

    spot_final = image_final.numpy()[image_final.shape[0] // 2]

    mi.set_variant('scalar_rgb')
    rfilter = mi.scalar_rgb.load_dict({'type': 'box'})
    image_init_np = np.array(mi.Bitmap(image_init_np).resample(resolution, rfilter))
    spot_init = image_init_np[resolution[0] // 2]

    image_final_np = np.zeros_like(image_init_np)
    low_idx = image_final_np.shape[0] // 2 - image_init.shape[0] // 2
    high_idx = low_idx + image_init.shape[0]
    image_final_np[low_idx:high_idx, low_idx:high_idx, :] = np.array(image_final)

    show_image(ax[2], image_init_np / (0.2 * np.max(image_init_np)), f'Initial spot size: {spot_size_init:.3e} mm')
    show_image(ax[3], image_final_np / (0.2 * np.max(image_final_np)), f'Final spot size: {spot_size_final:.3e} mm')

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
    ax[1].set_xlim([low_idx, high_idx])

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

