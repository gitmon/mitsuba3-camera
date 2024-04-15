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

template <typename Float, typename Spectrum>
class DispersiveMaterial {
    public:
        MI_IMPORT_TYPES()
        DispersiveMaterial(std::string name, float cauchy_A, float cauchy_B) : 
        m_name(name), m_cauchy(cauchy_A, cauchy_B), m_use_cauchy(true) {}

        DispersiveMaterial(std::string name, std::vector<std::pair<float, float>> sellmeier_terms) : 
        m_name(name), m_use_cauchy(false) {
            m_sellmeier_B = Vector3f(
                sellmeier_terms[0].first, 
                sellmeier_terms[1].first, 
                sellmeier_terms[2].first);
            m_sellmeier_C = Vector3f(
                sellmeier_terms[0].second, 
                sellmeier_terms[1].second, 
                sellmeier_terms[2].second);
        }

        DispersiveMaterial(std::string name, Vector3f sellmeier_B, Vector3f sellmeier_C) : 
        m_name(name), m_sellmeier_B(sellmeier_B), m_sellmeier_C(sellmeier_C), m_use_cauchy(false) { }

        Float compute_ior(const Ray3f& ray) const {
            if constexpr (!is_spectral_v<Spectrum>) {
                // if not rendering in spectral mode, return the "nominal" IOR 
                // (computed for a standard wavelength, 589.3nm)
                return compute_ior(Float(0.5893f));
            } else {
                // in spectral mode, each ray carries a *vector* of wavelengths.
                // for dispersion calculations, we take just the first wavelength.
                return compute_ior(0.001f * ray.wavelengths[0]);
            }
        }

        Float compute_ior(Float wavelength) const {
            return dr::select(m_use_cauchy, compute_ior_cauchy(wavelength), compute_ior_sellmeier(wavelength));
        }

        Float compute_abbe_number() const {
            return (compute_ior(Float(0.58756f)) - 1.0f) / (compute_ior(Float(0.4861f)) - compute_ior(Float(0.6563f)));
        }

        std::string get_name() const {
            return m_name;
        }

        std::string to_string() const {
            using string::indent;

            std::ostringstream oss;

            if (m_use_cauchy) {
                oss << "DispersiveMaterial[" << std::endl
                    << "  model = Cauchy, " << std::endl
                    << "  A0 = " << m_cauchy.x() << "," << std::endl
                    << "  B0 = " << m_cauchy.y() << std::endl;
            } else {
                oss << "DispersiveMaterial[" << std::endl
                    << "  model = Sellmeier, " << std::endl
                    << " Term " << 0 << ": " << std::endl
                    << "  B = " << indent(m_sellmeier_B.x())  << "," << std::endl
                    << "  C = " << indent(m_sellmeier_C.x()) << "," << std::endl
                    << " Term " << 0 << ": " << std::endl
                    << "  B = " << indent(m_sellmeier_B.y())  << "," << std::endl
                    << "  C = " << indent(m_sellmeier_C.y()) << "," << std::endl
                    << " Term " << 0 << ": " << std::endl
                    << "  B = " << indent(m_sellmeier_B.z())  << "," << std::endl
                    << "  C = " << indent(m_sellmeier_C.z()) << "," << std::endl;
            }
            oss << "]";
            return oss.str();
        }

    private:
        std::string m_name;
        Vector2f m_cauchy;
        Vector3f m_sellmeier_B, m_sellmeier_C;
        bool m_use_cauchy;

        Float compute_ior_cauchy(Float wavelength) const {
            // n = A + B / lbda_sq
            return m_cauchy.x() + m_cauchy.y() / dr::sqr(wavelength);
        }

        Float compute_ior_sellmeier(Float wavelength) const {
            // n ** 2 = 1.0 + sum(Bi * lbda_sq / (lbda_sq - Ci))
            Float wavelength_sq = dr::sqr(wavelength);
            Float ior = 1.f;
            ior += m_sellmeier_B.x() * wavelength_sq / (wavelength_sq - m_sellmeier_C.x());
            ior += m_sellmeier_B.y() * wavelength_sq / (wavelength_sq - m_sellmeier_C.y());
            ior += m_sellmeier_B.z() * wavelength_sq / (wavelength_sq - m_sellmeier_C.z());
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

template <typename Float, typename Spectrum>
class LensInterface {
    public:
        MI_IMPORT_TYPES()
        LensInterface(Float element_radius, Float z_intercept, DispersiveMaterial<Float, Spectrum> left_material, DispersiveMaterial<Float, Spectrum> right_material) : 
        m_z_intercept(z_intercept), m_element_radius(element_radius), m_left_material(left_material), m_right_material(right_material) {}

        virtual ~LensInterface() = default;

        virtual Interaction3f intersect(const Ray3f &ray) const = 0;

        virtual Normal3f normal(const Point3f &p) const = 0;

        Float get_radius() const {
            return m_element_radius;
        }

        Float get_z() const {
            return m_z_intercept;
        }

        virtual void offset_along_axis(Float delta) {
            m_z_intercept += delta;
        }

        virtual std::tuple<Ray3f, Mask> compute_interaction(const Ray3f &ray) const {
            Interaction3f si = intersect(ray);
            
            // if no intersection, early termination
            Mask active = si.is_valid();

            // reject intersection if it lies outside the lens' radius
            active &= (dr::sqr(si.p.x()) + dr::sqr(si.p.y())) < dr::sqr(m_element_radius);

            Float ext_ior = m_left_material.compute_ior(ray);
            Float int_ior = m_right_material.compute_ior(ray);

            // could replace with `Frame3f::cos_theta(si.wi)` if `si` were a SurfaceInteraction (equipped with local/shading frame)
            Float cos_theta_i = dr::dot(-ray.d, si.n);
            Float eta = int_ior / ext_ior;

            // fresnel() handles the int_ior/ext_ior swap 
            auto [r, cos_theta_t, eta_it, eta_ti] = fresnel(cos_theta_i, eta);

            // if internal reflection occurs, early termination
            active &= (r <= dr::OneMinusEpsilon<Float>);

            // get refraction direction in *global frame* (not `si`'s shading frame)
            Vector3f d_out = refract(-ray.d, si.n, cos_theta_t, eta_ti);
            Ray3f next_ray = dr::zeros<Ray3f>();
            dr::masked(next_ray, active) = Ray3f(si.p, d_out, dr::Largest<Float>, si.time, si.wavelengths);

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

        virtual std::vector<Point3f> draw_surface(int num_points, bool start_from_axis) const {
            std::vector<Point3f> points = {};
            Float radius = 0.f;
            for (int i = 0; i < num_points; ++i) {
                if (start_from_axis) {
                    radius = (m_element_radius * i) / (num_points - 1);
                }
                else {
                    radius = (m_element_radius * (num_points - 1 - i)) / (num_points - 1);
                }
                Ray3f ray(Point3f(radius, 0.f, m_z_intercept - 1.0f), Vector3f(0.f, 0.f, 1.f));
                Interaction3f si = intersect(ray);
                // TODO?
                // assert(si.is_valid());
                Point3f p_intersect = si.p;
                points.push_back(p_intersect);
            }
            return points;
        }

        std::string get_left_material() const {
            return m_left_material.get_name();
        }

        std::string get_right_material() const {
            return m_right_material.get_name();
        }

    protected:
        Float m_z_intercept;
        Float m_element_radius;
    private:
        DispersiveMaterial<Float, Spectrum> m_left_material, m_right_material;
};

template <typename Float, typename Spectrum>
class SpheroidLens final : public LensInterface<Float, Spectrum> {
    public:
        MI_IMPORT_TYPES()
        SpheroidLens(Float curvature_radius, Float aperture_radius, Float z_intercept, 
        DispersiveMaterial<Float, Spectrum> left_material, DispersiveMaterial<Float, Spectrum> right_material) : 
        LensInterface<Float, Spectrum>(aperture_radius, z_intercept, left_material, right_material), m_curvature_radius(curvature_radius) {
            m_center = Point3f(0.0, 0.0, LensInterface<Float, Spectrum>::m_z_intercept + m_curvature_radius);

            // sign convention: convex = positive radius, concave = negative
            m_is_convex = m_curvature_radius > 0.f;
        }

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

            // if discriminant is negative, no intersection
            Mask active = discriminant >= Float(0.0);
            if (dr::none_or<false>(active)) {       // TODO: this is how early exit is handled in `interaction.h`
                return si;
            }

            Float sqrt_disc = dr::sqrt(discriminant);
            Float near_t = center_proj - sqrt_disc;
            Float far_t = center_proj + sqrt_disc;

            active = far_t >= Float(0.0);
            if (dr::none_or<false>(active)) {       // TODO: this is how early exit is handled in `interaction.h`
                return si;
            }

            // ray.o is either inside the sphere, or in front of it.
            Float t_intersect;
            t_intersect = dr::select(m_is_convex ^ (ray.d.z() < 0.f),
                dr::select(near_t >= Float(0.f), near_t, dr::Infinity<Float>),
                far_t);

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

        void offset_along_axis(Float delta) override {
            LensInterface<Float, Spectrum>::m_z_intercept += delta;
            m_center = Point3f(m_center.x(), m_center.y(), LensInterface<Float, Spectrum>::m_z_intercept + m_curvature_radius);
        }

        std::string to_string() const override {
            using string::indent;
            std::ostringstream oss;

            oss << "SpheroidLens[" << std::endl
                << "  z_intercept = " << LensInterface<Float, Spectrum>::m_z_intercept << "," << std::endl
                << "  radius = " << m_curvature_radius << "," << std::endl
                << "  is_convex = " << m_is_convex << "," << std::endl
                << "]";
            return oss.str();
        }
    private:
        Float m_curvature_radius;
        Point3f m_center;
        Mask m_is_convex;
};


template <typename Float, typename Spectrum>
class PlanoLens : public LensInterface<Float, Spectrum> {
    public:
        MI_IMPORT_TYPES()
        PlanoLens(Normal3f normal, Float aperture_radius, Float z_intercept, 
        DispersiveMaterial<Float, Spectrum> left_material, DispersiveMaterial<Float, Spectrum> right_material) : 
        LensInterface<Float, Spectrum>(aperture_radius, z_intercept, left_material, right_material), m_normal(normal) {
            m_param = m_normal.z() * LensInterface<Float, Spectrum>::m_z_intercept;
        }

        PlanoLens(Float aperture_radius, Float z_intercept, 
        DispersiveMaterial<Float, Spectrum> left_material, DispersiveMaterial<Float, Spectrum> right_material) : 
        LensInterface<Float, Spectrum>(aperture_radius, z_intercept, left_material, right_material) {
            m_normal = Normal3f(0.f, 0.f, -1.0f);
            m_param = m_normal.z() * LensInterface<Float, Spectrum>::m_z_intercept;
        }

        Interaction3f intersect(const Ray3f &ray) const override {
            Interaction3f si = dr::zeros<Interaction3f>();
            si.time = ray.time;
            si.wavelengths = ray.wavelengths;

            // no-intersection case: ray.d is perpendicular to m_normal
            Float n_dot_d = dr::dot(m_normal, ray.d);
            Mask active = dr::abs(n_dot_d) >= dr::Epsilon<Float>;
            if (dr::none_or<false>(active)) {
                return si;
            }

            Float t = (m_param - dr::dot(m_normal, ray.o)) / n_dot_d;
            active = t >= Float(0.0);
            if (dr::none_or<false>(active)) {
                return si;
            }

            Point3f p_surface = ray(t);
            si.t = t;
            si.p = p_surface;
            si.n = normal(p_surface);

            return si;
        }

        Normal3f normal(const Point3f &p) const override {
            // TODO: direction depends on whether ray is entering/leaving medium?
            // CONVENTION: interior is -z, exterior is +z
            return m_normal;
        }

        void offset_along_axis(Float delta) override {
            LensInterface<Float, Spectrum>::m_z_intercept += delta;
            m_param = m_normal.z() * LensInterface<Float, Spectrum>::m_z_intercept;
        }

        std::string to_string() const override {
            using string::indent;
            std::ostringstream oss;

            oss << "PlanarLensInterface[" << std::endl
                << "  z_intercept = " << LensInterface<Float, Spectrum>::m_z_intercept << "," << std::endl
                << "  normal = " << m_normal << "," << std::endl
                << "]";
            return oss.str();
        }
    private:
        Normal3f m_normal;
        Float m_param;
};

template <typename Float, typename Spectrum>
class ApertureStop final : public PlanoLens<Float, Spectrum> {
    public:
        MI_IMPORT_TYPES()
        ApertureStop(Float aperture_radius, Float z_intercept, 
            DispersiveMaterial<Float, Spectrum> air_material) : 
        PlanoLens<Float, Spectrum>(aperture_radius, z_intercept, air_material, air_material) { }

        // Interaction3f intersect(const Ray3f &ray) const override {
        //     Interaction3f si = dr::zeros<Interaction3f>();
        //     si.time = ray.time;
        //     si.wavelengths = ray.wavelengths;

        //     // no-intersection case: ray.d is perpendicular to m_normal
        //     Float n_dot_d = dr::dot(m_normal, ray.d);
        //     Mask active = dr::abs(n_dot_d) >= dr::Epsilon<Float>;
        //     if (dr::none_or<false>(active)) {
        //         return si;
        //     }

        //     Float t = (m_param - dr::dot(m_normal, ray.o)) / n_dot_d;
        //     active = t >= Float(0.0);
        //     if (dr::none_or<false>(active)) {
        //         return si;
        //     }

        //     Point3f p_surface = ray(t);
        //     si.t = t;
        //     si.p = p_surface;
        //     si.n = normal(p_surface);

        //     return si;
        // }

        // Normal3f normal(const Point3f &p) const override {
        //     // TODO: direction depends on whether ray is entering/leaving medium?
        //     // CONVENTION: interior is -z, exterior is +z
        //     return m_normal;
        // }

        // void offset_along_axis(Float delta) override {
        //     LensInterface<Float, Spectrum>::m_z_intercept += delta;
        //     m_param = m_normal.z() * LensInterface<Float, Spectrum>::m_z_intercept;
        // }

        // implements a "no-op" for compute_interaction: if the ray is valid, it simply passes through
        // the aperture stop with its direction unchanged
        std::tuple<Ray3f, Mask> compute_interaction(const Ray3f &ray) const override {
            Interaction3f si = this->intersect(ray);
            
            // if no intersection, early termination
            // if (!intersected) { return false; }
            Mask active = si.is_valid();

            // reject intersection if it lies outside the stop radius
            // custom aperture shapes can also be implemented here
            active &= (dr::sqr(si.p.x()) + dr::sqr(si.p.y())) < dr::sqr(PlanoLens<Float, Spectrum>::m_element_radius);

            // create a new ray in the same direction; no refraction
            Ray3f next_ray = dr::zeros<Ray3f>();
            dr::masked(next_ray, active) = Ray3f(si.p, ray.d, dr::Largest<Float>, si.time, si.wavelengths);

            return { next_ray, active };
        }

        // overridden method for drawing the aperture: draw the *negation* of the
        // element area (i.e. lines' extent is [R, R + t] rather than [0, R])
        std::vector<Point3f> draw_surface(int num_points, bool start_from_axis) const override {
            std::vector<Point3f> points = {};
            Float radius = 0.f;
            Float stop_radius = PlanoLens<Float, Spectrum>::m_element_radius;
            for (int i = 0; i < num_points; ++i) {
                if (start_from_axis) {
                    radius = stop_radius + (0.2f * stop_radius * i) / (num_points - 1);
                }
                else {
                    radius = stop_radius + (0.2f * stop_radius * (num_points - 1 - i)) / (num_points - 1);
                }
                Ray3f ray(Point3f(radius, 0.f, PlanoLens<Float, Spectrum>::m_z_intercept - 1.0f), Vector3f(0.f, 0.f, 1.f));
                Interaction3f si = this->intersect(ray);
                // TODO?
                // assert(si.is_valid());
                Point3f p_intersect = si.p;
                points.push_back(p_intersect);
            }
            return points;
        }

        std::string to_string() const override {
            using string::indent;
            std::ostringstream oss;

            oss << "ApertureStop[" << std::endl
                << "  z_intercept = " << PlanoLens<Float, Spectrum>::m_z_intercept << "," << std::endl
                << "]";
            return oss.str();
        }
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

        std::string lens_type = props.get<std::string>("lens_design", "singlet");

        m_aperture_radius = props.get<ScalarFloat>("aperture_radius");

        if (dr::all(dr::eq(m_aperture_radius, 0.f))) {
            Log(Warn, "Can't have a zero aperture radius -- setting to %f", dr::Epsilon<Float>);
            m_aperture_radius = dr::Epsilon<Float>;
        }

        if (m_to_world.scalar().has_scale())
            Throw("Scale factors in the camera-to-world transformation are not allowed!");

        update_camera_transforms();

        m_needs_sample_3 = true;

        run_tests();
        
        Float object_distance, focal_length, lens_diameter;
        object_distance = props.get<Float>("object_distance", 6.0f);
        focal_length    = props.get<Float>("focal_length",    0.05f);
        lens_diameter   = props.get<Float>("lens_diameter",   0.01f);

        if (lens_type == "singlet") {
            build_thin_lens(object_distance, focal_length, lens_diameter / 2);
        } else if (lens_type == "doublet") {
            build_right_doublet_lens(object_distance, focal_length / 2, lens_diameter / 2);
        } else if (lens_type == "flipped_doublet") {
            build_flipped_doublet_lens(object_distance, focal_length / 2, lens_diameter / 2);
        } else if (lens_type == "tessar") {
            build_tessar_lens(object_distance);
        } else {
            build_thin_lens(object_distance, focal_length, lens_diameter / 2);
        }

        // Float object_distance = 0.5f;
        // Float focal_length = 0.05f;
        // Float lens_diameter = 0.005f;
        // build_thin_lens(object_distance, focal_length, lens_diameter / 2);

        // Float lens_diameter = 0.02f;
        // build_plano_lens(Float(0.02f), Float(0.01f), lens_diameter / 2, lens_diameter / 10);

        // compute_exit_pupil_bounds();

    }

    void loop_v1() const {
        float xmin = 0.0f, xmax = 0.5f;
        int N = 20;
        float dx = (xmax - xmin) / ((float) N - 1.f);

        for (int i = 0; i < N; ++i) {
            Vector3f o(i * dx, 0.f, 0.f);
            Vector3f d(0.f, 0.f, 1.f);
            Ray3f ray = Ray3f(o, d);
            draw_ray_from_film(ray);
            std::cout << "\n";
        }
        std::cout << std::endl;
    }


    void build_thin_lens(Float object_distance, Float curvature_radius, Float lens_radius) {
        // place the film plane at the image formation distance `xi` away from the lens
        // equivalently, keep the film plane at z=0 and move the lens to `z_intercept` = `xi`

        // clamp to ensure a real image is formed
        Float distance = dr::maximum(object_distance, 4.001f * curvature_radius);

        // set the lens position using the *thin lens* equation; use this to validate
        // that `focus_thick_lens()` is behaving correctly. 
        Float z_intercept = 0.5f * distance * (1.f - dr::sqrt(1.f - 4.f * curvature_radius / distance));
        Float thickness = 2.f * curvature_radius * (1.f - dr::sqrt(1.f - (lens_radius / curvature_radius) * (lens_radius / curvature_radius)));

        DispersiveMaterial<Float, Spectrum> air_material = DispersiveMaterial<Float, Spectrum>("Air", 1.000277f, 0.0f);
        DispersiveMaterial<Float, Spectrum> glass_material = DispersiveMaterial<Float, Spectrum>("NBK7", 1.5046f, 0.00420f);
        auto lens1 = new SpheroidLens<Float, Spectrum>(curvature_radius, lens_radius, z_intercept, air_material, glass_material);
        m_interfaces.push_back(lens1);
        auto lens2 = new SpheroidLens<Float, Spectrum>(-curvature_radius, lens_radius, z_intercept + thickness, glass_material, air_material);
        m_interfaces.push_back(lens2);

        // m_lens_aperture_z = z_intercept;
        // m_lens_aperture_radius = lens_radius;

        // // get a (conservative) estimate of the lens' total extent. This is used to launch
        // // rays from the outside world towards the lens body.
        // m_lens_terminal_z = m_interfaces.back()->get_z() + dr::abs(curvature_radius);

        // Float delta = focus_thick_lens(distance);
        // Float tmp = -distance / (1.f - distance / curvature_radius);

        // std::cout << "Adjustment from focus_thick_lens() (should be close to zero): " << -delta << std::endl;


        Float delta = focus_thick_lens(distance);
        for (const auto &interface : m_interfaces) {
            interface->offset_along_axis(-delta);
        }

        // draw_cross_section(16);

        m_lens_aperture_z = m_interfaces.front()->get_z();
        m_lens_aperture_radius = m_interfaces.front()->get_radius();
        m_lens_terminal_z = m_interfaces.back()->get_z() + dr::abs(curvature_radius);

    }


    void build_flipped_doublet_lens(Float object_distance, Float R, Float lens_radius) {
        // NOTE: our doublet focal length formula only applies if the two glasses have the same index!!!
        Float focal_length = 2.0f * R;
        Float distance = dr::maximum(object_distance, 4.001f * focal_length);
        Float z_intercept = 0.5f * distance * (1.f - dr::sqrt(1.f - 4.f * focal_length / distance));
        
        // z_intercept += 0.0201023f;
        Float thickness = 2.f * R * (1.f - dr::sqrt(1.f - (lens_radius / R) * (lens_radius / R)));

        DispersiveMaterial<Float, Spectrum> air = DispersiveMaterial<Float, Spectrum>("Air", 1.000277f, 0.0f);
        // crown glass for convex lens; N-BK7
        DispersiveMaterial<Float, Spectrum> glass_1 = DispersiveMaterial<Float, Spectrum>("NBK7", 1.5046f, 0.00420f);
        // flint glass for concave lens; N-SF5
        // std::vector<std::pair<float, float>> terms = {
        //                                     std::pair(1.52481889f, 0.01125475600f), 
        //                                     std::pair(0.187085527f, 0.0588995392f), 
        //                                     std::pair(1.427290150f, 129.1416750f)};
        // DispersiveMaterial<Float, Spectrum> glass_2 = DispersiveMaterial<Float, Spectrum>(terms);
        // fake N-BK7 with a tweaked abbe number, to cancel `glass_1` and form a functional achromat
        DispersiveMaterial<Float, Spectrum> glass_2 = DispersiveMaterial<Float, Spectrum>("mod-NBK7", 1.5046f, 0.00860948454f);

        std::cout << "Abbe numbers: V1 = " << glass_1.compute_abbe_number() << ", V2 = " << glass_2.compute_abbe_number() << "\n";
        std::cout << "We should have V1 = 2 * V2.\n";
        auto elem1 = new SpheroidLens<Float, Spectrum>(R, lens_radius, z_intercept, air, glass_1);
        m_interfaces.push_back(elem1);
        auto elem2 = new SpheroidLens<Float, Spectrum>(-R, lens_radius, z_intercept + thickness, glass_1, glass_2);
        m_interfaces.push_back(elem2);
        auto elem3 = new PlanoLens<Float, Spectrum>(Normal3f(0.f,0.f,-1.f), lens_radius, z_intercept + 2.f * thickness, glass_2, air);
        m_interfaces.push_back(elem3);
        // auto elem4 = new ApertureStop<Float, Spectrum>(0.1f * lens_radius, z_intercept + 3.f * thickness, air);
        // m_interfaces.push_back(elem4);

        Float delta = focus_thick_lens(distance);
        std::cout << "Pre-focus: " << delta << "\n";
        // draw_cross_section(16);

        for (const auto &interface : m_interfaces) {
            std::cout << interface->get_z() << ", ";
            interface->offset_along_axis(-delta);
            std::cout << interface->get_z() << "\n";
        }

        m_lens_aperture_z = m_interfaces.front()->get_z();
        m_lens_aperture_radius = m_interfaces.front()->get_radius();
        // get a (conservative) estimate of the lens' total extent. This is used to launch
        // rays from the outside world towards the lens body.
        m_lens_terminal_z = m_interfaces.back()->get_z() + dr::abs(R);

        // draw_cross_section(16);

        delta = focus_thick_lens(distance);

        std::cout << "Adjustment from focus_thick_lens() (should be close to zero): " << -delta << std::endl;
    }

    void build_right_doublet_lens(Float object_distance, Float R, Float lens_radius) {
        Float focal_length = 2.0f * R;
        Float distance = dr::maximum(object_distance, 4.001f * focal_length);
        Float z_intercept = 0.5f * distance * (1.f - dr::sqrt(1.f - 4.f * focal_length / distance));
        Float thickness = 2.f * R * (1.f - dr::sqrt(1.f - (lens_radius / R) * (lens_radius / R)));

        DispersiveMaterial<Float, Spectrum> air = DispersiveMaterial<Float, Spectrum>("Air", 1.000277f, 0.0f);
        DispersiveMaterial<Float, Spectrum> glass_1 = DispersiveMaterial<Float, Spectrum>("NBK7", 1.5046f, 0.00420f);
        DispersiveMaterial<Float, Spectrum> glass_2 = DispersiveMaterial<Float, Spectrum>("mod-NBK7", 1.5046f, 0.00860948454f);

        auto elem3 = new PlanoLens<Float, Spectrum>(Normal3f(0.f,0.f,-1.f), lens_radius, z_intercept, air, glass_2);
        m_interfaces.push_back(elem3);
        auto elem2 = new SpheroidLens<Float, Spectrum>(R, lens_radius, z_intercept + thickness, glass_2, glass_1);
        m_interfaces.push_back(elem2);
        auto elem1 = new SpheroidLens<Float, Spectrum>(-R, lens_radius, z_intercept + 2.f * thickness, glass_1, air);
        m_interfaces.push_back(elem1);
        auto elem0 = new ApertureStop<Float, Spectrum>(1.f * lens_radius, z_intercept + 3.f * thickness, air);
        m_interfaces.push_back(elem0);

        Float delta = focus_thick_lens(distance);
        for (const auto &interface : m_interfaces) {
            interface->offset_along_axis(-delta);
        }

        // draw_cross_section(16);

        m_lens_aperture_z = m_interfaces.front()->get_z();
        m_lens_aperture_radius = m_interfaces.front()->get_radius();
        m_lens_terminal_z = m_interfaces.back()->get_z() + dr::abs(R);
    }


    void build_plano_lens(Float z_intercept, Float thickness, Float lens_radius, Float aperture_radius) {
        DispersiveMaterial<Float, Spectrum> air = DispersiveMaterial<Float, Spectrum>("Air", 1.000277f, 0.0f);
        DispersiveMaterial<Float, Spectrum> glass = DispersiveMaterial<Float, Spectrum>("NBK7", 1.5046f, 0.00420f);

        auto elem1 = new PlanoLens<Float, Spectrum>(lens_radius, z_intercept, air, glass);
        m_interfaces.push_back(elem1);
        auto elem2 = new PlanoLens<Float, Spectrum>(lens_radius, z_intercept + thickness, glass, air);
        m_interfaces.push_back(elem2);
        auto elem3 = new ApertureStop<Float, Spectrum>(aperture_radius, z_intercept + 2.f * thickness, air);
        m_interfaces.push_back(elem3);

        m_lens_aperture_z = m_interfaces.front()->get_z();
        m_lens_aperture_radius = m_interfaces.front()->get_radius();
        m_lens_terminal_z = m_interfaces.back()->get_z() + 1.0f;

        // draw_cross_section(16);
    }

    void build_tessar_lens(Float object_distance) {
        // Parameters from:
        // https://henryquach.org/tessar.html

        DispersiveMaterial<Float, Spectrum> air = DispersiveMaterial<Float, Spectrum>("Air", 1.000277f, 0.0f);
        DispersiveMaterial<Float, Spectrum> NLAK9 = 
            DispersiveMaterial<Float, Spectrum>("NLAK9", 
            Vector3f(1.462319050, 0.344399589, 1.155083720), 
            Vector3f(0.007242702, 0.0243353131, 85.46868680));
        DispersiveMaterial<Float, Spectrum> K10 = 
            DispersiveMaterial<Float, Spectrum>("K10", 
            Vector3f(1.156870820, 0.064262544, 0.872376139), 
            Vector3f(0.008094243, 0.0386051284, 104.74773000));
        DispersiveMaterial<Float, Spectrum> F2 = 
            DispersiveMaterial<Float, Spectrum>("F2", 
            Vector3f(1.397570370, 0.159201403, 1.268654300), 
            Vector3f(0.009959061, 0.0546931752, 119.24834600));

        // -
        // 0. convert mm -> m
        // 1. reverse order of elements
        // 2. start summing thickness from 0
        // 3. flip sign of radii
        // 4. radius = diameter / 2
        // 5. auto-fill materials

        Float z_pos = 0.0f;
        Float t;

        // surface8
        t = 0.001f * 86.917f;
        z_pos += t;
        m_interfaces.push_back(new SpheroidLens<Float, Spectrum>(0.001f * 43.567f, 0.001f * 16.f / 2, z_pos, air, NLAK9));

        // surface7
        t = 0.001f * 9.941f;
        z_pos += t;
        m_interfaces.push_back(new SpheroidLens<Float, Spectrum>(0.001f * -45.344f, 0.001f * 16.f / 2, z_pos, NLAK9, K10));

        // surface6
        t = 0.001f * 2.286f;
        z_pos += t;
        m_interfaces.push_back(new SpheroidLens<Float, Spectrum>(0.001f * 86.620f, 0.001f * 16.f / 2, z_pos, K10, air));

        // surface5
        t = 0.001f * 1.999f;
        z_pos += t;
        m_interfaces.push_back(new ApertureStop<Float, Spectrum>(0.001f * 9.3f / 2, z_pos, air));

        // surface4
        t = 0.001f * 2.289f;
        z_pos += t;
        m_interfaces.push_back(new SpheroidLens<Float, Spectrum>(0.001f * -31.297f, 0.001f * 12.f / 2, z_pos, air, F2));

        // surface3
        t = 0.001f * 2.290f;
        z_pos += t;
        m_interfaces.push_back(new SpheroidLens<Float, Spectrum>(0.001f * 63.028f, 0.001f * 12.f / 2, z_pos, F2, air));

        // surface2
        t = 0.001f * 2.286f;
        z_pos += t;
        m_interfaces.push_back(new SpheroidLens<Float, Spectrum>(0.001f * 296.111f, 0.001f * 18.f / 2, z_pos, air, NLAK9));

        // surface1
        t = 0.001f * 3.567f;
        z_pos += t;
        m_interfaces.push_back(new SpheroidLens<Float, Spectrum>(0.001f * -35.034f, 0.001f * 18.f / 2, z_pos, NLAK9, air));

        // draw_cross_section(16);

        Float delta = focus_thick_lens(object_distance);
        for (const auto &interface : m_interfaces) {
            interface->offset_along_axis(-delta);
        }

        std::cout << "Fine focus adjustment: " << delta << std::endl;

        m_lens_aperture_z = m_interfaces.front()->get_z();
        m_lens_aperture_radius = m_interfaces.front()->get_radius();
        m_lens_terminal_z = m_interfaces.back()->get_z() + dr::abs(m_interfaces.back()->get_radius());
    }



    std::tuple<Ray3f, Mask> trace_ray_from_film(const Ray3f &ray) const {
        Mask active = true;
        Ray3f curr_ray(ray);
        // TODO: dr::loop method causes a segfault :(
        // size_t lens_id = 0;

        // dr::Loop<Mask> loop("trace", active, lens_id, curr_ray);
        // std::cout << "======== NEW RAY ========" << std::endl;
        // while(loop(active)) {
        //     auto [next_ray, next_active] = m_interfaces.at(lens_id)->compute_interaction(curr_ray);
        //     // curr_ray = next_ray;
        //     // curr_ray = Ray3f(next_ray.o, next_ray.d, next_ray.maxt, next_ray.time, next_ray.wavelengths);
        //     std::cout << "==== Index: " << lens_id << " ====\n";
        //     // std::cout << curr_ray << ",\t";
        //     std::cout << next_active << ",\t";
        //     std::cout << next_ray << "\n\n";
    
        //     // std::cout << "B1\n";

        //     curr_ray.o = next_ray.o;
        //     curr_ray.d = next_ray.d;
        //     curr_ray.maxt = next_ray.maxt;
        //     curr_ray.time = next_ray.time;
        //     curr_ray.wavelengths = next_ray.wavelengths;
        //     // std::cout << "B2\n";
        //     lens_id += 1;
        //     // std::cout << "B3\n";
        //     active &= next_active && (lens_id < m_interfaces.size());
        //     // std::cout << "B4\n";
        //     // active &= (lens_id < m_interfaces.size());
        //     std::cout << "==== index complete ====\n";
        // }
        // std::cout << "======== END RAY ========" << std::endl;

        // std::cout << "B5\n";

        // std::cout << "======== NEW RAY ========" << std::endl;
        for (const auto &interface : m_interfaces) {
            // TODO: is it better to mask?
            // TODO: actually, replace this with a dr::loop! then while-loop through
            // all the lens elements and add `&& active` to the conditional. rays that
            // fail will terminate early and have active == false
                        // ray_ = interface->compute_interaction(ray_, active);

            // std::cout << "==== Index: ====\n";
            auto [next_ray, next_active] = interface->compute_interaction(curr_ray);

            // std::cout << next_active << ",\t";
            // std::cout << next_ray << "\n\n";

            active &= next_active;

            // std::cout << "==== index complete ====\n";
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
        // std::cout << "======== END RAY ========" << std::endl;
        
        return { curr_ray, active };
    }


    // void compute_exit_pupil_bounds() {
    //     int num_segments = 64;
    //     int rays_per_segment = 1024 * 1024;
    //     m_exit_pupil_bounds.resize(num_segments);
    //     Float diagonal = dr::norm(m_film->get_physical_size());

    //     Float rear_radius = m_interfaces[0]->get_radius() * 1.5f;
    //     Float rear_z = m_interfaces[0]->get_z();
    //     BoundingBox2f rear_bounds(
    //         Point2f(-rear_radius, -rear_radius),
    //         Point2f( rear_radius,  rear_radius));

    //     // TODO: dr::loop?
    //     for (int i = 0; i < num_segments; ++i) {
    //         // TODO: initialization
    //         BoundingBox2f pupil_bound(Point3f(0.0f));
    //         pupil_bound.reset();
    //         Float r0 = i * diagonal / num_segments;
    //         Float r1 = (i + 1) * diagonal / num_segments;

    //         // initialize and launch rays
    //         for (int j = 0; j < rays_per_segment; ++j) {
    //             Point3f p_film(dr::lerp(r0, r1, (j + 0.5f) / rays_per_segment), 
    //                 0.0f, 
    //                 0.0f);
    //             // TODO: replace m_sampler with radical_inverse_2(j)
    //             // (using the sampler here is a waste of random numbers. we just need
    //             // a well-distributed 2d grid of points, not doing MC integration)
    //             Point3f p_rear(
    //                 dr::lerp(-rear_radius, rear_radius, m_sampler->next_1d()),
    //                 dr::lerp(-rear_radius, rear_radius, m_sampler->next_1d()),
    //                 rear_z);
                
    //             Ray3f ray(p_rear, dr::normalize(Vector3f(p_rear - p_film)));

    //             // if (pupil_bound.contains(p_rear[x,y]) || trace_ray_from_film()) {
    //             //     pupil_bound.expand(p_rear[x,y]);
    //             // }


    //             // auto [ray_out, active_out] = trace_ray_from_film(ray);
    //             // Vector3f d_out(ray_out.d);
    //             // active &= active_out;
                
    //         }

    //         m_exit_pupil_bounds[i] = pupil_bound;
    //     }

    // }

    void draw_cross_section(int num_points) const {
        size_t vtx_idx = 0;
        std::vector<Point3f> points = {};
        std::vector<Point2i> edges = {};
        bool new_element;

        for (size_t surface_id = 0; surface_id < m_interfaces.size(); ++surface_id) {
            auto s = m_interfaces[surface_id];
            bool left_is_air =  string::to_lower(s->get_left_material()) == "air";
            bool right_is_air = string::to_lower(s->get_right_material()) == "air";

            std::vector<Point3f> p_list;
            if (left_is_air) {
                // cases:
                //  1. air->air interface; aperture
                //  2. air->glass interface
                new_element = true;
                p_list = s->draw_surface(num_points, true);
            } else {
                // cases:
                //  3. glass->air interface
                //  4. glass->glass interface
                new_element = false;
                p_list = s->draw_surface(num_points, false);
            }

            // add points to the output
            for (size_t i = 0; i < p_list.size(); ++i) {
                points.push_back(p_list[i]);
                vtx_idx = points.size() - 1;
                if (new_element && i == 0) {
                    continue;
                }
                // connect curr point to previous point in the list
                edges.push_back(Point2i(vtx_idx - 1, vtx_idx));
            }

            // for glass-glass interface, draw the interface a second time
            if (!left_is_air && !right_is_air) {
                p_list = s->draw_surface(num_points, true);
                new_element = true;
                for (size_t i = 0; i < p_list.size(); ++i) {
                    points.push_back(p_list[i]);
                    vtx_idx = points.size() - 1;
                    if (new_element && i == 0) {
                        continue;
                    }
                    edges.push_back(Point2i(vtx_idx - 1, vtx_idx));
                }
            }
        }

        // print points and edges
        std::cout << "points = [";
        for (const auto& p : points) {
            std::cout << p << ",\n";
        }
        std::cout << "]";
        // print points and edges
        std::cout << "\n\nedges = [";
        for (const auto& e : edges) {
            std::cout << e << ",\n";
        }
        std::cout << "]";

    }


    // Traces a ray from the world side of the lens to the film side. The input
    // `ray` is assumed to be represented in camera space.
    std::tuple<Ray3f, Mask> trace_ray_from_world(const Ray3f &ray) const {
        Mask active = true;
        Ray3f curr_ray(ray);
        // TODO: switch to dr::loop when trace_ray_from_film() issue is fixed

        // std::cout << "======== NEW RAY ========" << std::endl;
        for (int lens_id = m_interfaces.size() - 1; lens_id >= 0; lens_id--) {
            // std::cout << "==== Index: ====\n";
            auto [next_ray, next_active] = m_interfaces[lens_id]->compute_interaction(curr_ray);

            // std::cout << next_active << ",\t";
            // std::cout << next_ray << "\n\n";

            active &= next_active;

            // std::cout << "==== index complete ====\n";
            if (dr::none_or<false>(active)) {
                break;
            }

            curr_ray = next_ray;
        }
        // std::cout << "======== END RAY ========" << std::endl;
        
        return { curr_ray, active };
    }

    // Test the two tracing functions against each other; according to 
    // reciprocity, we should have ray_out == ray_in.
    Mask test_trace_ray_from_world(const Ray3f &ray) const {
        Ray3f film_ray = Ray3f(ray, dr::Infinity<Float>);
        auto [world_ray, active] = trace_ray_from_film(film_ray);

        // if the film ray doesn't even reach the world side, there's
        // no point in checking trace_ray_from_world()
        if (dr::none_or<false>(active)) {
            std::cout << "A\n";
            return true;
        }

        auto [out_ray, active_] = trace_ray_from_world(world_ray.reverse());

        // the film->world trace was performed successfully. If the backwards
        // trace doesn't work, something must be wrong with either algorithm
        if (dr::any_or<true>(!active_)) {
            std::cout << "B\n";
            return false;
        }

        // Trace the lens[0] intersection back to the aperture plane
        out_ray.o = out_ray((m_lens_aperture_z - out_ray.o.z()) * dr::rcp(out_ray.d.z()));

        // check that init_ray and backward(forward(init_ray)) are equal
        out_ray = out_ray.reverse();
        // Float err = 0.5f * (dr::norm(out_ray.o - film_ray.o) + dr::norm(out_ray.d - film_ray.d));

        std::cout << "Err_pos: " << dr::norm(out_ray.o - film_ray.o) 
                  << ", err_dir: " << dr::norm(dr::cross(out_ray.d, film_ray.d));

        Float tol = math::RayEpsilon<Float> * m_interfaces.size();
        Mask passed = dr::norm(out_ray.o - film_ray.o) < tol && 
               dr::norm(dr::cross(out_ray.d, film_ray.d)) < tol;

        std::cout << ", passed: " << passed << "\n";

        return passed;
    }



    void draw_ray_from_film(const Ray3f &ray) const {
        Mask active = true;
        Ray3f curr_ray(ray);

        // Vector3f d_film = dr::normalize(curr_ray.o - p_film);

        // std::cout   << p_film.x() 
        //     << ", " << p_film.y() 
        //     << ", " << p_film.z() 
        //     << ", " << d_film.x()
        //     << ", " << d_film.y()
        //     << ", " << d_film.z()
        //     << ", " << active;
        // std::cout   << ",\t";

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


    void draw_ray_from_world(const Ray3f &ray) const {
        Mask active = true;
        Ray3f curr_ray(ray);

        for (int lens_id = m_interfaces.size() - 1; lens_id >= 0; lens_id--) {
            std::cout   << curr_ray.o.x() 
                << ", " << curr_ray.o.y() 
                << ", " << curr_ray.o.z() 
                << ", " << curr_ray.d.x()
                << ", " << curr_ray.d.y()
                << ", " << curr_ray.d.z()
                << ", " << active;
            std::cout   << ",\t";


            // std::cout << "Lens id: " << lens_id << "\t";
            auto [next_ray, active_] = m_interfaces[lens_id]->compute_interaction(curr_ray);
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
        std::cout   << ",\t";

        // Trace the lens[0] intersection back to the aperture plane
        curr_ray.o = curr_ray((m_lens_aperture_z - curr_ray.o.z()) * dr::rcp(curr_ray.d.z()));

        std::cout   << curr_ray.o.x() 
            << ", " << curr_ray.o.y() 
            << ", " << curr_ray.o.z() 
            << ", " << curr_ray.d.x()
            << ", " << curr_ray.d.y()
            << ", " << curr_ray.d.z()
            << ", " << active;

        std::cout   << std::endl;
    }

    // Compute the axial positions of the principal plane and paraxial
    // focus from either the object (world) or image (film) sides of 
    // the lens. `start_ray` and `end_ray` are the ray segments before/
    // after tracing through the lens respectively.
    // NOTE: for simplicity, we consider rays that are launched from the
    // x-axis only (rather than arbitrary (x,y,0)). 
    void compute_cardinal_points(const Ray3f& start_ray, const Ray3f& end_ray, 
                                Float& z_p, Float& z_f) const {
        Float t_focus = -end_ray.o.x() / end_ray.d.x();
        z_f = end_ray(t_focus).z();
        Float t_plane = (start_ray.o.x() - end_ray.o.x()) / end_ray.d.x();
        z_p = end_ray(t_plane).z();
    }

    void compute_thick_lens_approximation(Float& back_plane_z, Float& back_focal_length, 
                                          Float& front_plane_z, Float& front_focal_length) const {
        // set radial distance for estimating the paraxial quantities
        Float r = 0.001f * m_film->get_physical_size().x();

        // object (world)-side quantities
        Float obj_plane, obj_focus;
        Ray3f obj_ray = Ray3f(
            Point3f(r, 0.f, m_lens_terminal_z + 1.0f), 
            Vector3f(0.f, 0.f, -1.f),
            0.0f,
            Wavelength(589.3f));

        auto [obj_end_ray, active] = trace_ray_from_world(obj_ray);

        if (dr::none_or<false>(active)) {
            Throw("compute_thick_lens_approximation: world ray was not transmitted through lens!");
        }

        compute_cardinal_points(obj_ray, obj_end_ray, obj_plane, obj_focus);
        back_plane_z = obj_plane;
        // back_focal_length = obj_focus - obj_plane;
        back_focal_length = obj_plane - obj_focus;
        std::cout << "\nback_plane: " << back_plane_z << ", focal: " << obj_focus << std::endl;

        // image (film)-side quantities
        Float img_plane, img_focus;
        Ray3f img_ray = Ray3f(
            Point3f(r, 0.f, m_lens_aperture_z - 1.0f), 
            Vector3f(0.f, 0.f, 1.f),
                        0.0f,
            Wavelength(589.3f));

        auto [img_end_ray, active_] = trace_ray_from_film(img_ray);

        draw_ray_from_film(img_ray);

        if (dr::none_or<false>(active_)) {
            Throw("compute_thick_lens_approximation: film ray was not transmitted through lens!");
        }

        compute_cardinal_points(img_ray, img_end_ray, img_plane, img_focus);
        front_plane_z = img_plane;
        front_focal_length = img_focus - img_plane;
        std::cout << "\nfront_plane: " << front_plane_z << ", focal: " << img_focus << std::endl;
    }

    Float focus_thick_lens(Float focus_distance) {
        // NOTE: the film-to-object distance must be at least four focal lengths: otherwise,
        // the image becomes virtual and nothing will be on the film plane (numerically, this
        // also results in a sqrt(-x) error).
        // To resolve this issue, we clamp the focus distance to be at least 4 * f_img
        Float p_img, f_img, p_obj, f_obj;
        compute_thick_lens_approximation(p_img, f_img, p_obj, f_obj);
        Float tmp = dr::maximum(focus_distance, 4.01f * f_img) - p_obj;
        Float delta = 0.5f * (p_img - tmp + dr::sqrt(dr::sqr(p_img + tmp) - 4 * f_img * (p_img + tmp)));
        return delta;
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

        // std::cout << "Wave weight: " << wav_weight << "\n";

        // STAGE 1: FILM SAMPLING

        // Compute the sample position on the near plane (local camera space).
        // Point3f film_p = m_sample_to_camera *
        //                 Point3f(position_sample.x(), position_sample.y(), 0.f);
        // ------------------------

        // Compute the sample position on the near plane. For RealisticCamera, this is 
        // the physical location of a point on the film, expressed in local camera space. 
        // The film occupies [-xmax, xmax] x [-ymax, ymax] x [0,0]. Meanwhile, 
        // `position_sample` is a uniform 2D sample distributed on [0,1]^2.
        Point3f film_p = m_sample_to_film *
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
        // Point3f focus_p = film_p * (m_focus_distance / film_p.z());

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

        // Set up the film->pupil ray. The ray starts at `film_p` and is directed
        //  along the vector connecting `film_p` and `aperture_p`
        // std::cout << "A\n";
        Vector3f d = dr::normalize(Vector3f(aperture_p - film_p));
        ray.o = film_p;
        ray.d = d;

        // std::cout << ray.o << std::endl;

        // Trace the ray through the lens
        // std::cout << "B: tracing pixel, " << position_sample << ", " << aperture_sample << "\n";
        auto [ray_out, active_out] = trace_ray_from_film(ray);
        // std::cout << "C\n";
        Vector3f d_out(ray_out.d);
        // std::cout << "D\n";

        // std::cout << active_out << ", " << active << ", " << wav_weight << std::endl;

        active &= active_out;

        // std::cout << "E\n";
        // Kill rays that don't get through the lens
        dr::masked(wav_weight, !active) = dr::zeros<Spectrum>();
        // std::cout << d_out << ",\t" << wav_weight << ",\t" << active << "\n";

        // draw_ray_from_film(ray);
        // std::cout << ", ";
        // draw_ray_from_world(ray_out.reverse());

        // std::cout << "F\n";

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

        // std::cout << "G\n";

        // std::cout << "Reciprocity test: " << test_trace_ray_from_world(ray) << std::endl;

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
    //     Point3f film_p = m_sample_to_camera *
    //                     Point3f(position_sample.x(), position_sample.y(), 0.f);

    //     // Aperture position
    //     Point2f tmp = m_aperture_radius * warp::square_to_uniform_disk_concentric(aperture_sample);
    //     Point3f aperture_p(tmp.x(), tmp.y(), 0.f);

    //     // Sampled position on the focal plane
    //     Float f_dist = m_focus_distance / film_p.z();
    //     Point3f focus_p   = film_p          * f_dist,
    //             focus_p_x = (film_p + m_dx) * f_dist,
    //             focus_p_y = (film_p + m_dy) * f_dist;

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
    std::vector<BoundingBox2f> m_exit_pupil_bounds;
    Float m_aperture_radius;
    Float m_normalization;
    Float m_x_fov;
    Vector3f m_dx, m_dy;
    // std::vector<std::unique_ptr<LensInterface<Float, Spectrum>>> m_interfaces;
    std::vector<LensInterface<Float, Spectrum>*> m_interfaces;
    Float m_lens_aperture_z, m_lens_aperture_radius, m_lens_terminal_z;




    // temporary place to put tests
    void test_materials() {
        auto mat = DispersiveMaterial<Float, Spectrum>("BK7", 1.5046f, 0.00420f);
        assert(dr::allclose(mat.compute_ior(2.3254f), 1.505376701));
        assert(dr::allclose(mat.compute_ior(0.5893f), 1.516694179));
        assert(dr::allclose(mat.compute_ior(0.3126f), 1.547580488));


        std::vector<std::pair<float, float>> terms = {
            std::make_pair(1.03961212f, 0.006000699f),
            std::make_pair(0.231792344f, 0.020017914f),
            std::make_pair(1.01046945f, 103.560653f),
        };

        auto mat2 = DispersiveMaterial<Float, Spectrum>("NBK7", terms);
        assert(dr::allclose(mat2.compute_ior(2.3254f), 1.489211725));
        assert(dr::allclose(mat2.compute_ior(0.5893f), 1.516727674));
        assert(dr::allclose(mat2.compute_ior(0.3126f), 1.548606931));
    }

    void run_tests() {
        test_materials();
        // TODO
        // test_trace_ray_from_world(const Ray3f &ray);
    }
};

MI_IMPLEMENT_CLASS_VARIANT(RealisticLensCamera, ProjectiveCamera)
MI_EXPORT_PLUGIN(RealisticLensCamera, "Realistic Lens Camera");
NAMESPACE_END(mitsuba)
