#include <mitsuba/render/sensor.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/bbox.h>
#include <mitsuba/core/warp.h>

NAMESPACE_BEGIN(mitsuba)

/**!
// TODO

.. _sensor-thinlens:

Perspective camera with a thin lens (:monosp:`thinlens`)
--------------------------------------------------------

.. pluginparameters::
 :extra-rows: 8

 * - to_world
   - |transform|
   - Specifies an optional camera-to-world transformation.
     (Default: none (i.e. camera space = world space))
   - |exposed|

 * - aperture_radius
   - |float|
   - Denotes the radius of the camera's aperture in scene units.
   - |exposed|

 * - focus_distance
   - |float|
   - Denotes the world-space distance from the camera's aperture to the focal plane.
     (Default: :monosp:`0`)
   - |exposed|

 * - focal_length
   - |string|
   - Denotes the camera's focal length specified using *35mm* film equivalent units.
     See the main description for further details. (Default: :monosp:`50mm`)

 * - fov
   - |float|
   - An alternative to :monosp:`focal_length`: denotes the camera's field of view in degrees---must be
     between 0 and 180, excluding the extremes.

 * - fov_axis
   - |string|
   - When the parameter :monosp:`fov` is given (and only then), this parameter further specifies
     the image axis, to which it applies.

     1. :monosp:`x`: :monosp:`fov` maps to the :monosp:`x`-axis in screen space.
     2. :monosp:`y`: :monosp:`fov` maps to the :monosp:`y`-axis in screen space.
     3. :monosp:`diagonal`: :monosp:`fov` maps to the screen diagonal.
     4. :monosp:`smaller`: :monosp:`fov` maps to the smaller dimension
        (e.g. :monosp:`x` when :monosp:`width` < :monosp:`height`)
     5. :monosp:`larger`: :monosp:`fov` maps to the larger dimension
        (e.g. :monosp:`y` when :monosp:`width` < :monosp:`height`)

     The default is :monosp:`x`.

 * - near_clip, far_clip
   - |float|
   - Distance to the near/far clip planes. (Default: :monosp:`near_clip=1e-2` (i.e. :monosp:`0.01`)
     and :monosp:`far_clip=1e4` (i.e. :monosp:`10000`))
   - |exposed|

 * - srf
   - |spectrum|
   - Sensor Response Function that defines the :ref:`spectral sensitivity <explanation_srf_sensor>`
     of the sensor (Default: :monosp:`none`)

 * - x_fov
   - |float|
   - Denotes the camera's field of view in degrees along the horizontal axis.
   - |exposed|

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/sensor_thinlens_small_aperture.jpg
   :caption: The material test ball viewed through a perspective thin lens camera. (:monosp:`aperture_radius=0.1`)
.. subfigure:: ../../resources/data/docs/images/render/sensor_thinlens.jpg
   :caption: The material test ball viewed through a perspective thin lens camera. (:monosp:`aperture_radius=0.2`)
.. subfigend::
   :label: fig-thinlens

This plugin implements a simple perspective camera model with a thin lens
at its circular aperture. It is very similar to the
:ref:`perspective <sensor-perspective>` plugin except that the extra lens element
permits rendering with a specifiable (i.e. non-infinite) depth of field.
To configure this, it has two extra parameters named :monosp:`aperture_radius`
and :monosp:`focus_distance`.

By default, the camera's field of view is specified using a 35mm film
equivalent focal length, which is first converted into a diagonal field
of view and subsequently applied to the camera. This assumes that
the film's aspect ratio matches that of 35mm film (1.5:1), though the
parameter still behaves intuitively when this is not the case.
Alternatively, it is also possible to specify a field of view in degrees
along a given axis (see the :monosp:`fov` and :monosp:`fov_axis` parameters).

The exact camera position and orientation is most easily expressed using the
:monosp:`lookat` tag, i.e.:

.. tabs::
    .. code-tab:: xml

        <sensor type="thinlens">
            <float name="fov" value="45"/>
            <transform name="to_world">
                <!-- Move and rotate the camera so that looks from (1, 1, 1) to (1, 2, 1)
                    and the direction (0, 0, 1) points "up" in the output image -->
                <lookat origin="1, 1, 1" target="1, 2, 1" up="0, 0, 1"/>
            </transform>

            <!-- Focus on the target -->
            <float name="focus_distance" value="1"/>
            <float name="aperture_radius" value="0.1"/>

            <!-- film -->
            <!-- sampler -->
        </sensor>

    .. code-tab:: python

        'type': 'thinlens',
        'fov': 45,
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[1, 1, 1],
            target=[1, 2, 1],
            up=[0, 0, 1]
        ),
        'focus_distance': 1.0,
        'aperture_radius': 0.1,
        'film_id': {
            'type': '<film_type>',
            # ...
        },
        'sampler_id': {
            'type': '<sampler_type>',
            # ...
        }

 */

// TODO: is the template needed? 
// TODO: need to handle multiple wavelengths? since `Ray3f` can contain multiple?
template <typename Float>
class DispersiveMaterial {
    public:
        DispersiveMaterial(float cauchy_A, float cauchy_B) : m_cauchy_A(cauchy_A), m_cauchy_B(cauchy_B), m_use_cauchy(true) {}

        DispersiveMaterial(std::vector<std::pair<float, float>> sellmeier_terms) : m_sellmeier_terms(sellmeier_terms), m_use_cauchy(false) {}

        // Float compute_ior(Ray3f ray) const {
            // if (!is_spectral_v<Spectrum>) {
            //     // if not rendering in spectral mode, return the "nominal" IOR 
            //     // (computed for a standard wavelength, 589.3nm)
            //     return compute_ior();
            // } else {
            //     // in spectral mode, each ray carries a *vector* of wavelengths
            //     // because of hero wavelength sampling. it doesn't make sense to
            //     // compute multiple IORs (and thus find multiple paths) per ray,
            //     // so let's just take the first wavelength (I guess???)
            //     // TODO
            //     return compute_ior(0.001 * ray.wavelengths[0]);
            // }
        // }

        Float compute_ior(Float wavelength) const {
            return dr::select(m_use_cauchy, compute_ior_cauchy(wavelength), compute_ior_sellmeier(wavelength));
        }

        Float compute_ior() const {
            return compute_ior(Float(0.5893f));
        }

        std::string to_string() const {
            using string::indent;

            std::ostringstream oss;

            if (m_use_cauchy) {
                oss << "DispersiveMaterial[" << std::endl
                    << "  model = Cauchy, " << std::endl
                    << "  A0 = " << m_cauchy_A << "," << std::endl
                    << "  B0 = " << m_cauchy_B << std::endl;
            } else {
                int n_terms = m_sellmeier_terms.size();

                oss << "DispersiveMaterial[" << std::endl
                    << "  model = Sellmeier, " << std::endl;
                
                for (int i = 0; i < n_terms; i++) {
                    auto term = m_sellmeier_terms[i];
                    oss << " Term " << i << ": " << std::endl
                        << "  B = " << indent(term.first)  << "," << std::endl
                        << "  C = " << indent(term.second) << "," << std::endl;
                }
            }
            oss << "]";
            return oss.str();
        }

    private:
        const char* name;
        float m_cauchy_A = 0.f;
        float m_cauchy_B = 0.f;
        std::vector<std::pair<float, float>> m_sellmeier_terms;
        bool m_use_cauchy;

        Float compute_ior_cauchy(Float wavelength) const {
            // n = A + B / lbda_sq
            return m_cauchy_A + m_cauchy_B / dr::sqr(wavelength);
        }

        Float compute_ior_sellmeier(Float wavelength) const {
            // n ** 2 = 1.0 + sum(Bi * lbda_sq / (Ci - lbda_sq))
            Float wavelength_sq = dr::sqr(wavelength);
            Float ior = 1.f;    // TODO: dr::ones?

            for (const auto &term : m_sellmeier_terms) {
                ior += term.first * wavelength_sq / (wavelength_sq - term.second);
            }
            ior = dr::sqrt(ior);
            return ior;
        }
};

// // static DispersiveMaterial<float> lens_materials[] = {
//     // { "air",     DispersiveMaterial(1.000277f, 0.0f) },
//     // { "n-bk7",   DispersiveMaterial({
//     //         {1.03961212f, 0.006000699f},
//     //         {0.231792344f, 0.020017914f},
//     //         {1.01046945f, 103.560653f},
//     //     })},
// // };

// TODO: are the templates needed? the IMPORT_TYPES() fails without it
template <typename Float, typename Spectrum>
class LensInterface {
    public:
    MI_IMPORT_TYPES()
        LensInterface(float aperture_radius, float z_intercept, DispersiveMaterial<Float> int_material, DispersiveMaterial<Float> ext_material) : 
        m_z_intercept(z_intercept), m_aperture_radius(aperture_radius), m_int_material(int_material), m_ext_material(ext_material) {}

        virtual ~LensInterface() = default;

        virtual Interaction3f intersect(const Ray3f &ray) const = 0;

        virtual Normal3f normal(const Point3f &p) const = 0;

        float get_radius() const {
            return m_aperture_radius;
        }

        float get_z() const {
            return m_z_intercept;
        }

        std::tuple<Ray3f, Mask> compute_interaction(const Ray3f &ray) {
            Interaction3f si = intersect(ray);
            
            // if no intersection, early termination
            // if (!intersected) { return false; }
            Mask active = si.is_valid();

            // reject intersection if it lies outside the lens' radius
            // if (dr::sqr(si.p.x()) + dr::sqr(si.p.y()) >= dr::sqr(m_aperture_radius)) { return false; }
            active &= (dr::sqr(si.p.x()) + dr::sqr(si.p.y()) < dr::sqr(m_aperture_radius));

            // TODO: compute IOR using the ray's wavelength
            // int_ior = m_int_material->compute_ior(ray);
            Float int_ior = m_int_material.compute_ior();
            Float ext_ior = m_ext_material.compute_ior();

            // could replace with `Frame3f::cos_theta(si.wi)` if `si` were a SurfaceInteraction (equipped with local/shading frame)
            Float cos_theta_i = dr::dot(-ray.d, si.n);
            Float eta = int_ior / ext_ior;

            // fresnel() handles the int_ior/ext_ior swap 
            auto [r, cos_theta_t, eta_it, eta_ti] = fresnel(cos_theta_i, eta);

            // std::cout << "c_i: " << cos_theta_i << ", c_o: " << cos_theta_t << ", eta: " << eta_ti << "\n";
            // std::cout << ", eta: " << eta_ti << ", ";
            // std::cout << "nnn: " << si.n.x() << ", " << si.n.y() << ", " << si.n.z() << "\n";

            // if internal reflection occurs, early termination
            // if (r >= dr::OneMinusEpsilon<Float>) { return false; }
            active &= (r <= dr::OneMinusEpsilon<Float>);

            // get refraction direction in *global frame* (not `si`'s shading frame)
            Vector3f d_out = refract(-ray.d, si.n, cos_theta_t, eta_ti);
            Ray3f next_ray = dr::zeros<Ray3f>();
            dr::masked(next_ray, active) = si.spawn_ray(d_out);

            return { next_ray, active };
        }

        virtual std::string to_string() const {
            using string::indent;
            std::ostringstream oss;

            oss << "LensInterface[" << std::endl
                << "  z_intercept = " << m_z_intercept << "," << std::endl
                << "]";
            return oss.str();
        }

    protected:
        float m_z_intercept;
    private:
        float m_aperture_radius;
        DispersiveMaterial<Float> m_int_material, m_ext_material;
};

template <typename Float, typename Spectrum>
class SphericalLensInterface final : public LensInterface<Float, Spectrum> {
    public:
        MI_IMPORT_TYPES()
        SphericalLensInterface(float curvature_radius, float aperture_radius, float z_intercept, 
        DispersiveMaterial<Float> int_material, DispersiveMaterial<Float> ext_material) : 
        LensInterface<Float, Spectrum>(aperture_radius, z_intercept, int_material, ext_material), m_curvature_radius(curvature_radius) {
            m_center = Point3f(0.0, 0.0, LensInterface<Float, Spectrum>::m_z_intercept + m_curvature_radius);

            // sign convention: convex = positive radius, concave = negative
            m_is_convex = m_curvature_radius > 0.f;
        }

        // TODO: should bools be Masks?
        // TODO: use math::solve_quadratic for quadratic solve
        Interaction3f intersect(const Ray3f &ray) const override {
            Interaction3f si = dr::zeros<Interaction3f>();
            si.time = ray.time;
            si.wavelengths = ray.wavelengths;

            // compute sphere position in the ray's "frame"
            Point3f p_center_local = (Point3f) (m_center - ray.o);
            Float center_proj = dr::dot(p_center_local, ray.d);
            Float perp_projection = dr::norm(p_center_local - center_proj * ray.d);
            Float discriminant = dr::sqr(m_curvature_radius) - dr::sqr(perp_projection);

            // // TODO: handling `if` clauses
            // // if discriminant is negative, no intersection
            Mask active = discriminant >= Float(0.0);
            if (dr::none_or<false>(active)) {       // TODO: this is how early exit is handled in `interaction.h`
                // std::cout << "D < 0, no intersection\n";
                return si;
            }

            Float sqrt_disc = dr::sqrt(discriminant);
            Float near_t = center_proj - sqrt_disc;
            Float far_t = center_proj + sqrt_disc;

            // from `sphere.cpp`
            // Sphere doesn't intersect with the segment on the ray
            // Mask out_bounds = far_t < Float(0.0);
            // if (dr::none_or<false>(active)) {       // TODO: this is how early exit is handled in `interaction.h`
            //     // std::cout << "sphere behind ray, no intersection\n";
            //     return si;
            // }

            active = far_t >= Float(0.0);
            if (dr::none_or<false>(active)) {       // TODO: this is how early exit is handled in `interaction.h`
                // std::cout << "sphere behind ray, no intersection\n";
                return si;
            }

            // ray.o is either inside the sphere, or in front of it.
            Float t_intersect;
            if (m_is_convex) {
                // convex case
                // we are only testing intersection with the convex/near half of the sphere.
                // if `near_t` is positive, the intersection is valid; otherwise, no intersection
                // active &= near_t >= Float(0.0);
                // t_intersect = dr::select(active, near_t, dr::Infinity<Float>);
                t_intersect = dr::select(near_t >= Float(0.f), near_t, dr::Infinity<Float>);
            } 
            else {
                // concave case
                // always take `far_t`. from the earlier bounds check, we know that `far_t` 
                // is already positive, so it's a valid intersection
                // t_intersect = dr::select(active, far_t, dr::Infinity<Float>);
                t_intersect = far_t;
            }

            Point3f p_surface = ray(t_intersect);
            si.t = t_intersect;
            si.p = p_surface;
            si.n = normal(p_surface);

            return si;
        }

        Normal3f normal(const Point3f &p) const override {
            Normal3f normal = (Normal3f) dr::normalize(p - m_center);
            return dr::select(m_is_convex, normal, -normal);
        }

        std::string to_string() const override {
            using string::indent;
            std::ostringstream oss;

            oss << "SphericalLensInterface[" << std::endl
                << "  z_intercept = " << LensInterface<Float, Spectrum>::m_z_intercept << "," << std::endl
                << "  radius = " << m_curvature_radius << "," << std::endl
                << "  is_convex = " << m_is_convex << "," << std::endl
                << "]";
            return oss.str();
        }
    private:
        float m_curvature_radius;
        Point3f m_center;
        bool m_is_convex;
};


template <typename Float, typename Spectrum>
class RealisticLensCamera final : public ProjectiveCamera<Float, Spectrum> {
public:
    MI_IMPORT_BASE(ProjectiveCamera, m_to_world, m_needs_sample_3, m_film, m_sampler,
                    m_resolution, m_shutter_open, m_shutter_open_time, m_near_clip,
                    m_far_clip, m_focus_distance, sample_wavelengths)
    MI_IMPORT_TYPES()

    RealisticLensCamera(const Properties &props) : Base(props) {
        ScalarVector2i size = m_film->size();
        m_x_fov = (ScalarFloat) parse_fov(props, size.x() / (double) size.y());

        m_aperture_radius = props.get<ScalarFloat>("aperture_radius");

        if (dr::all(dr::eq(m_aperture_radius, 0.f))) {
            Log(Warn, "Can't have a zero aperture radius -- setting to %f", dr::Epsilon<Float>);
            m_aperture_radius = dr::Epsilon<Float>;
        }

        if (m_to_world.scalar().has_scale())
            Throw("Scale factors in the camera-to-world transformation are not allowed!");

        update_camera_transforms();

        m_needs_sample_3 = true;

        // auto mat = DispersiveMaterial<Float>(1.5046f, 0.00420f);
        // std::cout << mat.compute_ior(0.5893f) << std::endl;
        // std::cout << mat.to_string() << std::endl;

        // std::vector<std::pair<float, float>> terms = {
        //     std::make_pair(1.03961212f, 0.006000699f),
        //     std::make_pair(0.231792344f, 0.020017914f),
        //     std::make_pair(1.01046945f, 103.560653f),
        // };

        // auto mat2 = DispersiveMaterial<Float>(terms);
        // std::cout << mat2.compute_ior(0.5893f) << std::endl;
        // std::cout << mat2.to_string() << std::endl;

        // build_lens();
        float object_distance = 0.5f;
        float focal_length = 0.05f;
        float lens_diameter = 0.03f;
        build_thin_lens(object_distance, focal_length, lens_diameter / 2);

        // float r = 6.f;
        // float xmin, ymin, xmax, ymax;
        // xmin = ymin = 0.f;
        // xmax = ymax = r;
        // int N = 20;
        // float dx = (xmax - xmin) / ((float) N - 1.f);

        // for (int i = 0; i < N; ++i) {
        //     for (int j = 0; j < N; ++j) {
        //         Vector3f o(i * dx, j * dx, 0.f);
        //         Vector3f d(0.f, 0.f, 1.f);
        //         Ray3f ray = Ray3f(o, d);

        //         std::cout   << ray.o.x() 
        //             << ", " << ray.o.y() 
        //             << ", " << ray.o.z() 
        //             << ", " << ray.d.x()
        //             << ", " << ray.d.y()
        //             << ", " << ray.d.z();
        //         auto [ray_out, active] = trace_ray_through_lens(ray);
        //         std::cout   << "\n";
        //         // std::cout   << ray_out.o.x() << ", " 
        //         //             << ray_out.o.y() << ", " 
        //         //             << ray_out.o.z() << ", " 
        //         //             << ray_out.d.x() << ", "
        //         //             << ray_out.d.y() << ", "
        //         //             << ray_out.d.z() << "\n";
        //     }
        // }
        // std::cout << std::endl;
    }

    void build_thin_lens(float object_distance, float curvature_radius, float lens_radius) {
        // place the film plane at the image formation distance `xi` away from the lens
        // equivalently, keep the film plane at z=0 and move the lens to `z_intercept` = `xi`
        float z_intercept = object_distance / (1.f + object_distance / curvature_radius);
        float thickness = 2.f * curvature_radius * (1.f - std::sqrt(1.f - (lens_radius / curvature_radius) * (lens_radius / curvature_radius)));

        DispersiveMaterial<Float> air_material = DispersiveMaterial<Float>(1.0f, 0.0f);
        DispersiveMaterial<Float> glass_material = DispersiveMaterial<Float>(1.5f, 0.0f);
        auto lens1 = new SphericalLensInterface<Float, Spectrum>(curvature_radius, lens_radius, z_intercept, glass_material, air_material);
        m_interfaces.push_back(lens1);
        auto lens2 = new SphericalLensInterface<Float, Spectrum>(-curvature_radius, lens_radius, z_intercept + thickness, air_material, glass_material);
        m_interfaces.push_back(lens2);

        m_lens_aperture_z = z_intercept;
        m_lens_aperture_radius = lens_radius;

        float magnification = z_intercept / object_distance;
        float film_halfsize = std::max(m_film->get_physical_size().x(), m_film->get_physical_size().y());
        float approx_fov = dr::rad_to_deg(2.f * dr::atan((film_halfsize + lens_radius) / z_intercept));
        std::cout << "Lens thickness: " << thickness * 1000.f << " mm, \n";
        std::cout << "Thickness ratio: " << thickness / curvature_radius << ", \n";
        std::cout << "Lens position: " << z_intercept << " m, \n";
        std::cout << "Approx FOV: " << approx_fov << " deg., \n";
        std::cout << "Magnification: " << magnification << "\n";
    }

    void build_lens() {
        // float aperture_radius = 5.f;
        // float curvature_radius = 10.f;
        // float z_intercept = 0.1f;

        float aperture_radius = 0.001f;
        float curvature_radius = 1.f;
        float z_intercept = 0.02f;
        float thickness = 0.005f;

        // float aperture_radius = 0.03f;
        // float curvature_radius = 0.03f;
        // float z_intercept = 0.02f;
        // float thickness = 0.05f;
        m_lens_aperture_radius = 0.001f;

        // TODO: change to `new MyClass()` ? 
        DispersiveMaterial<Float> air_material = DispersiveMaterial<Float>(1.0f, 0.0f);
        DispersiveMaterial<Float> glass_material = DispersiveMaterial<Float>(1.5f, 0.0f);

        // SphericalLensInterface<Float, Spectrum> lens(curvature_radius, aperture_radius, z_intercept, air_material, glass_material);

        // auto lens = std::make_unique<SphericalLensInterface<Float, Spectrum>>(curvature_radius, aperture_radius, z_intercept, air_material, glass_material);
        // m_interfaces.push_back(std::move(lens));

        auto lens1 = new SphericalLensInterface<Float, Spectrum>(curvature_radius, aperture_radius, z_intercept, glass_material, air_material);
        m_interfaces.push_back(lens1);
        
        // auto lens2 = new SphericalLensInterface<Float, Spectrum>(-curvature_radius, aperture_radius, z_intercept + 2 * curvature_radius, air_material, glass_material);
        // m_interfaces.push_back(lens2);
        auto lens2 = new SphericalLensInterface<Float, Spectrum>(-curvature_radius, aperture_radius, z_intercept + thickness, air_material, glass_material);
        m_interfaces.push_back(lens2);

        // auto lens2 = new SphericalLensInterface<Float, Spectrum>(curvature_radius, aperture_radius, z_intercept + thickness, air_material, glass_material);
        // m_interfaces.push_back(lens2);

        for (const auto &interface : m_interfaces) {
            std::cout << interface->to_string() << std::endl;
        }

        // TODO: re-enable
        m_lens_aperture_z = m_interfaces[0]->get_z();
        // m_lens_aperture_radius = m_interfaces[0]->get_radius();
    }

    std::tuple<Ray3f, Mask> trace_ray_through_lens(const Ray3f &ray) const {
        Mask active = true;
        Ray3f curr_ray(ray);
        // UInt32 lens_id = 0;

        // dr::Loop<Mask> loop("trace", active, lens_id, curr_ray);
        // while(loop(active)) {
        //     auto [next_ray, next_active] = m_interfaces[lens_id]->compute_interaction(curr_ray);
        //     curr_ray = next_ray;
        //     lens_id += 1;
        //     active &= next_active && (lens_id < m_interfaces.size());
        // }


        // std::cout << active << ", ";
        for (const auto &interface : m_interfaces) {
            // TODO: is it better to mask?
            // TODO: actually, replace this with a dr::loop! then while-loop through
            // all the lens elements and add `&& active` to the conditional. rays that
            // fail will terminate early and have active == false
                        // ray_ = interface->compute_interaction(ray_, active);
            auto [next_ray, active_] = interface->compute_interaction(curr_ray);

            active &= active_;

            if (dr::none_or<false>(active)) {
                break;
            }

            curr_ray = next_ray;
            // std::cout   << ", " << ray_.o.x() 
            //             << ", " << ray_.o.y() 
            //             << ", " << ray_.o.z() 
            //             << ", " << ray_.d.x()
            //             << ", " << ray_.d.y()
            //             << ", " << ray_.d.z();
        }
        // std::cout << active << std::endl;
        // std::cout << ray_.o << ", " << ray_.d << std::endl;
        
        return { curr_ray, active };
    }


    void draw_ray_through_lens(const Ray3f &ray, const Point3f p_film) const {
        Mask active = true;
        Ray3f curr_ray(ray);

        Vector3f d_film = dr::normalize(curr_ray.o - p_film);

        std::cout   << p_film.x() 
            << ", " << p_film.y() 
            << ", " << p_film.z() 
            << ", " << d_film.x()
            << ", " << d_film.y()
            << ", " << d_film.z()
            << ", " << active;
        std::cout   << ",\t";

        for (const auto &interface : m_interfaces) {
            std::cout   << curr_ray.o.x() 
                << ", " << curr_ray.o.y() 
                << ", " << curr_ray.o.z() 
                << ", " << curr_ray.d.x()
                << ", " << curr_ray.d.y()
                << ", " << curr_ray.d.z()
                << ", " << active;
            std::cout   << ",\t";

            auto [next_ray, active_] = interface->compute_interaction(curr_ray);
            active &= active_;
            if (dr::none_or<false>(active)) { break; }
            curr_ray = next_ray;
        }
        std::cout   << curr_ray.o.x() 
            << ", " << curr_ray.o.y() 
            << ", " << curr_ray.o.z() 
            << ", " << curr_ray.d.x()
            << ", " << curr_ray.d.y()
            << ", " << curr_ray.d.z()
            << ", " << active;
        
        std::cout   << std::endl;
    }



    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("aperture_radius", m_aperture_radius, +ParamFlags::NonDifferentiable);
        callback->put_parameter("focus_distance",  m_focus_distance,  +ParamFlags::NonDifferentiable);
        callback->put_parameter("x_fov",           m_x_fov,           +ParamFlags::NonDifferentiable);
        callback->put_parameter("to_world",       *m_to_world.ptr(),  +ParamFlags::NonDifferentiable);
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        Base::parameters_changed(keys);
        if (keys.empty() || string::contains(keys, "to_world")) {
            if (m_to_world.scalar().has_scale())
                Throw("Scale factors in the camera-to-world transformation are not allowed!");
        }

        update_camera_transforms();
    }

    void update_camera_transforms() {
        m_film_to_sample = film_to_crop_transform(
            m_film->get_physical_size(), m_film->size(), 
            m_film->crop_size(), m_film->crop_offset()); 

        m_sample_to_film = m_film_to_sample.inverse();

        // TODO: deprecate the following

        m_camera_to_sample = perspective_projection(
            m_film->size(), m_film->crop_size(), m_film->crop_offset(),
            m_x_fov, Float(m_near_clip), Float(m_far_clip));

        m_sample_to_camera = m_camera_to_sample.inverse();

        // Position differentials on the near plane
        m_dx = m_sample_to_camera * Point3f(1.f / m_resolution.x(), 0.f, 0.f)
             - m_sample_to_camera * Point3f(0.f);
        m_dy = m_sample_to_camera * Point3f(0.f, 1.f / m_resolution.y(), 0.f)
             - m_sample_to_camera * Point3f(0.f);

        /* Precompute some data for importance(). Please
           look at that function for further details. */
        Point3f pmin(m_sample_to_camera * Point3f(0.f, 0.f, 0.f)),
                pmax(m_sample_to_camera * Point3f(1.f, 1.f, 0.f));

        m_image_rect.reset();
        m_image_rect.expand(Point2f(pmin.x(), pmin.y()) / pmin.z());
        m_image_rect.expand(Point2f(pmax.x(), pmax.y()) / pmax.z());
        m_normalization = 1.f / m_image_rect.volume();

        dr::make_opaque(m_camera_to_sample, m_sample_to_camera, m_dx, m_dy,
                        m_x_fov, m_image_rect, m_normalization);
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &position_sample,
                                          const Point2f &aperture_sample,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        auto [wavelengths, wav_weight] =
            sample_wavelengths(dr::zeros<SurfaceInteraction3f>(),
                               wavelength_sample,
                               active);
        Ray3f ray;
        ray.time = time;
        ray.wavelengths = wavelengths;

        // STAGE 1: FILM SAMPLING

        // Compute the sample position on the near plane (local camera space).
        // Point3f near_p = m_sample_to_camera *
        //                 Point3f(position_sample.x(), position_sample.y(), 0.f);
        // ------------------------

        // Compute the sample position on the near plane. For RealisticCamera, this is 
        // the physical location of a point on the film, expressed in local camera space. 
        // The film occupies [-xmax, xmax] x [-ymax, ymax] x [0,0]. Meanwhile, 
        // `position_sample` is a uniform 2D sample distributed on [0,1]^2.
        Point3f near_p = m_sample_to_film *
                        Point3f(position_sample.x(), position_sample.y(), 0.f);

        // STAGE 2: APERTURE SAMPLING

        // // Aperture position
        // Point2f tmp = m_aperture_radius * warp::square_to_uniform_disk_concentric(aperture_sample);
        // Point3f aperture_p(tmp.x(), tmp.y(), 0.f);
        // ------------------------

        // Sample the exit pupil
        Point2f tmp = m_lens_aperture_radius * warp::square_to_uniform_disk_concentric(aperture_sample);
        Point3f aperture_p(tmp.x(), tmp.y(), m_lens_aperture_z);
        // Point3f aperture_p(0.f, 0.f, m_lens_aperture_z);

        // STAGE 3: RAY SETUP

        // // Sampled position on the focal plane
        // Point3f focus_p = near_p * (m_focus_distance / near_p.z());

        // // Convert into a normalized ray direction; adjust the ray interval accordingly.
        // Vector3f d = dr::normalize(Vector3f(focus_p - aperture_p));
        // ray.o = m_to_world.value().transform_affine(aperture_p);
        // ray.d = m_to_world.value() * d;

        // Float inv_z = dr::rcp(d.z());
        // Float near_t = m_near_clip * inv_z,
        //       far_t  = m_far_clip * inv_z;
        // ray.o += ray.d * near_t;
        // ray.maxt = far_t - near_t;
        // ------------------------

        // Set up the film->pupil ray. The ray starts at `aperture_p` and is directed
        //  along the vector connecting `film_p` and `aperture_p`
        Vector3f d = dr::normalize(Vector3f(aperture_p - near_p));
        ray.o = aperture_p;
        ray.d = d;

        // std::cout << ray.o << std::endl;

        // Trace the ray through the lens
        auto [ray_out, active_out] = trace_ray_through_lens(ray);
        Vector3f d_out(ray_out.d);

        // std::cout << active_out << ", " << active << ", " << wav_weight << std::endl;

        active &= active_out;

        // Kill rays that don't get through the lens
        // TODO
        dr::masked(wav_weight, !active) = dr::zeros<Spectrum>();

        // draw_ray_through_lens(ray, near_p);
        // std::cout << d_out << ",\t" << wav_weight << ",\t" << active << "\n";


        // Convert ray_out from camera to world space
        // dr::masked(ray_out, active) = m_to_world.value() * ray_out;
        // ray_out = m_to_world.value() * ray_out;
        ray_out.o = m_to_world.value().transform_affine(ray_out.o);
        ray_out.d = m_to_world.value() * ray_out.d;
        // ------------------------

        // STAGE 4: POST-PROCESS
        // handle z-clipping
        // NOTE: the direction `d` in inv_z should be in the camera frame, i.e. before `m_to_world` is applied
        // TODO
        Float inv_z = dr::rcp(d_out.z());
        Float near_t = m_near_clip * inv_z,
              far_t  = m_far_clip * inv_z;
        ray_out.o += ray_out.d * near_t;
        ray_out.maxt = far_t - near_t;


        // std::cout << ray_out.o << ",\t" << ray_out.d << ",\t" << ray_out.maxt << std::endl;

        return { ray_out, wav_weight };
    }


    // // NOTE: can we remove this and fallback to the default `sample_ray_differential()` implementation
    // // in sensor.cpp?
    // // TODO: figure out the stuff about `METHOD_NAME` vs `METHOD_NAME_impl`
    // std::pair<RayDifferential3f, Spectrum>
    // sample_ray_differential_impl(Float time, Float wavelength_sample,
    //                              const Point2f &position_sample, const Point2f &aperture_sample,
    //                              Mask active) const {
    //     MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

    //     auto [wavelengths, wav_weight] =
    //         sample_wavelengths(dr::zeros<SurfaceInteraction3f>(),
    //                            wavelength_sample,
    //                            active);
    //     RayDifferential3f ray;
    //     ray.time = time;
    //     ray.wavelengths = wavelengths;

    //     // Compute the sample position on the near plane (local camera space).
    //     Point3f near_p = m_sample_to_camera *
    //                     Point3f(position_sample.x(), position_sample.y(), 0.f);

    //     // Aperture position
    //     Point2f tmp = m_aperture_radius * warp::square_to_uniform_disk_concentric(aperture_sample);
    //     Point3f aperture_p(tmp.x(), tmp.y(), 0.f);

    //     // Sampled position on the focal plane
    //     Float f_dist = m_focus_distance / near_p.z();
    //     Point3f focus_p   = near_p          * f_dist,
    //             focus_p_x = (near_p + m_dx) * f_dist,
    //             focus_p_y = (near_p + m_dy) * f_dist;

    //     // Convert into a normalized ray direction; adjust the ray interval accordingly.
    //     Vector3f d = dr::normalize(Vector3f(focus_p - aperture_p));

    //     ray.o = m_to_world.value().transform_affine(aperture_p);
    //     ray.d = m_to_world.value() * d;

    //     Float inv_z = dr::rcp(d.z());
    //     Float near_t = m_near_clip * inv_z,
    //           far_t  = m_far_clip * inv_z;
    //     ray.o += ray.d * near_t;
    //     ray.maxt = far_t - near_t;

    //     ray.o_x = ray.o_y = ray.o;

    //     ray.d_x = m_to_world.value() * dr::normalize(Vector3f(focus_p_x - aperture_p));
    //     ray.d_y = m_to_world.value() * dr::normalize(Vector3f(focus_p_y - aperture_p));
    //     ray.has_differentials = true;

    //     return { ray, wav_weight };
    // }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample,
                     Mask active) const override {
        // Transform the reference point into the local coordinate system
        Transform4f trafo = m_to_world.value();
        Point3f ref_p = trafo.inverse().transform_affine(it.p);

        // Check if it is outside of the clip range
        DirectionSample3f ds = dr::zeros<DirectionSample3f>();
        ds.pdf = 0.f;
        active &= (ref_p.z() >= m_near_clip) && (ref_p.z() <= m_far_clip);
        if (dr::none_or<false>(active))
            return { ds, dr::zeros<Spectrum>() };

        // Sample a position on the aperture (in local coordinates)
        Point2f tmp = warp::square_to_uniform_disk_concentric(sample) * m_aperture_radius;
        Point3f aperture_p(tmp.x(), tmp.y(), 0);

        // Compute the normalized direction vector from the aperture position to the referent point
        Vector3f local_d = ref_p - aperture_p;
        Float dist     = dr::norm(local_d);
        Float inv_dist = dr::rcp(dist);
        local_d *= inv_dist;

        // Compute importance value
        Float ct     = Frame3f::cos_theta(local_d),
              inv_ct = dr::rcp(ct);
        Point3f scr = m_camera_to_sample.transform_affine(
            aperture_p + local_d * (m_focus_distance * inv_ct));
        Mask valid = dr::all(scr >= 0.f) && dr::all(scr <= 1.f);
        Float value = dr::select(valid, m_normalization * inv_ct * inv_ct * inv_ct, 0.f);

        if (dr::none_or<false>(valid))
            return { ds, dr::zeros<Spectrum>() };

        ds.uv   = dr::head<2>(scr) * m_resolution;
        ds.p    = trafo.transform_affine(aperture_p);
        ds.d    = (ds.p - it.p) * inv_dist;
        ds.dist = dist;
        ds.n    = trafo * Vector3f(0.f, 0.f, 1.f);

        Float aperture_pdf = dr::rcp(dr::Pi<Float> * dr::sqr(m_aperture_radius));
        ds.pdf = dr::select(valid, aperture_pdf * dist * dist * inv_ct, 0.f);

        return { ds, Spectrum(value * inv_dist * inv_dist) };
    }


    ScalarBoundingBox3f bbox() const override {
        ScalarPoint3f p = m_to_world.scalar() * ScalarPoint3f(0.f);
        return ScalarBoundingBox3f(p, p);
    }

    std::string to_string() const override {
        using string::indent;

        std::ostringstream oss;
        oss << "RealisticLensCamera[" << std::endl
            << "  x_fov = " << m_x_fov << "," << std::endl
            << "  near_clip = " << m_near_clip << "," << std::endl
            << "  far_clip = " << m_far_clip << "," << std::endl
            << "  focus_distance = " << m_focus_distance << "," << std::endl
            << "  film = " << indent(m_film) << "," << std::endl
            << "  sampler = " << indent(m_sampler) << "," << std::endl
            << "  resolution = " << m_resolution << "," << std::endl
            << "  shutter_open = " << m_shutter_open << "," << std::endl
            << "  shutter_open_time = " << m_shutter_open_time << "," << std::endl
            << "  to_world = " << indent(m_to_world)  << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    Transform4f m_film_to_sample;
    Transform4f m_sample_to_film;
    Transform4f m_camera_to_sample;
    Transform4f m_sample_to_camera;
    BoundingBox2f m_image_rect;
    Float m_aperture_radius;
    Float m_normalization;
    Float m_x_fov;
    Vector3f m_dx, m_dy;
    // std::vector<std::unique_ptr<LensInterface<Float, Spectrum>>> m_interfaces;
    std::vector<LensInterface<Float, Spectrum>*> m_interfaces;
    float m_lens_aperture_z, m_lens_aperture_radius;
};

MI_IMPLEMENT_CLASS_VARIANT(RealisticLensCamera, ProjectiveCamera)
MI_EXPORT_PLUGIN(RealisticLensCamera, "Realistic Lens Camera");
NAMESPACE_END(mitsuba)
