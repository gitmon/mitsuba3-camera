#include <mitsuba/render/sensor.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/bbox.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/qmc.h>
#define TEST_EXIT_PUPIL false

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

        virtual void draw_surface(std::vector<Point3f> &points, std::vector<Normal3f> &normals, int num_points, bool start_from_axis) const {
            points.clear();
            normals.clear();
            Float radius = 0.f;
            for (int i = 0; i < num_points; ++i) {
                if (start_from_axis) {
                    radius = (m_element_radius * i) / (num_points - 1);
                }
                else {
                    radius = (m_element_radius * (num_points - 1 - i)) / (num_points - 1);
                }

                Ray3f ray(Point3f(radius, 0.f, m_z_intercept - 1.f), Vector3f(0.f, 0.f, 1.f));
                Interaction3f si = intersect(ray);
                Point3f p_intersect = si.p;
                Normal3f normal = si.n;
                points.push_back(p_intersect);
                normals.push_back(normal);
            }
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
        SpheroidLens(Float curvature_radius, Float element_radius, Float z_intercept, 
        DispersiveMaterial<Float, Spectrum> left_material, DispersiveMaterial<Float, Spectrum> right_material) : 
        LensInterface<Float, Spectrum>(element_radius, z_intercept, left_material, right_material), m_curvature_radius(curvature_radius) {
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
            if (dr::none_or<false>(active)) {
                return si;
            }

            Float sqrt_disc = dr::sqrt(discriminant);
            Float near_t = center_proj - sqrt_disc;
            Float far_t = center_proj + sqrt_disc;

            active &= far_t >= Float(0.0);
            if (dr::none_or<false>(active)) {
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
        PlanoLens(Normal3f normal, Float element_radius, Float z_intercept, 
        DispersiveMaterial<Float, Spectrum> left_material, DispersiveMaterial<Float, Spectrum> right_material) : 
        LensInterface<Float, Spectrum>(element_radius, z_intercept, left_material, right_material), m_normal(normal) {
            m_param = m_normal.z() * LensInterface<Float, Spectrum>::m_z_intercept;
        }

        PlanoLens(Float element_radius, Float z_intercept, 
        DispersiveMaterial<Float, Spectrum> left_material, DispersiveMaterial<Float, Spectrum> right_material) : 
        LensInterface<Float, Spectrum>(element_radius, z_intercept, left_material, right_material) {
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

        // implements a "no-op" for compute_interaction: if the ray is valid, it simply passes through
        // the aperture stop with its direction unchanged
        std::tuple<Ray3f, Mask> compute_interaction(const Ray3f &ray) const override {
            Interaction3f si = this->intersect(ray);
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
        virtual void draw_surface(std::vector<Point3f> &points, std::vector<Normal3f> &normals, int num_points, bool start_from_axis) const override {
            points.clear();
            normals.clear();
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
                Point3f p_intersect = si.p;
                points.push_back(p_intersect);
                normals.push_back(si.n);
            }
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
class AsphericalLens final : public LensInterface<Float, Spectrum> {
    public:
        MI_IMPORT_TYPES()
        AsphericalLens(Float curvature_radius, Float kappa, Float element_radius, Float z0, std::vector<Float> ai,
        DispersiveMaterial<Float, Spectrum> left_material, DispersiveMaterial<Float, Spectrum> right_material) : 
        LensInterface<Float, Spectrum>(element_radius, z0, left_material, right_material), 
        m_K(kappa) { 
            // curvature and [ai] are in units of millimeters; rescale using `m_r_scale` (mm),
            // which is `element_radius` converted from (m) -> (mm)
            
            float MM_TO_METERS = 0.001f;

            m_r_scale = 1.0f / MM_TO_METERS * element_radius;
            m_c = m_r_scale * dr::rcp(curvature_radius);
            m_ai.reserve(ai.size());
            for (size_t i = 0; i < ai.size(); ++i) {
                m_ai.push_back(dr::pow(m_r_scale, 2 * i + 3) * ai.at(i));
            }
        }

        Interaction3f intersect(const Ray3f &ray) const override {
            // perform intersection in length units of millimeters
            // (easier to handle asphere coefficients)
            Float TOL = dr::Epsilon<Float> * 10.f;

            Interaction3f si = dr::zeros<Interaction3f>();
            si.time = ray.time;
            si.wavelengths = ray.wavelengths;

            // try to compute an initial guess for the asphere intersection by intersecting
            // against the conic surface
            auto [t, valid] = intersect_conic(ray);

            // if the conic intersection doesn't work, intersect with the plane `z = z_intercept` instead
            if (dr::none_or<false>(valid)) {
                t = (LensInterface<Float, Spectrum>::m_z_intercept - ray.o.z()) * dr::rcp(ray.d.z());
            }

            Point3f p_curr = ray(t);
            Float  r2_curr = dr::sqr(p_curr.x()) + dr::sqr(p_curr.y());
            Float err = dr::Infinity<Float>;
            UInt32 itr = 0;

            Mask active = true;
            dr::Loop<Mask> loop("trace", t, active, p_curr, r2_curr, err, itr);

            // compute asphere intersection using newton's method
            while(loop(active)) {
                // build tangent plane on the asphere

                // dr::resume_grad<Float> scope(true, r2_curr);
                // std::cout << dr::grad_enabled(r2_curr) << std::endl;
                // dr::enable_grad(r2_curr);
                // dr::set_grad(r2_curr, 1.0);
                // std::cout << dr::grad_enabled(r2_curr) << std::endl;
                Float z_asph = eval_asph(r2_curr);
                // dr::forward_to(z_asph);
                // Float z_grad = dr::grad(z_asph);
                // dr::disable_grad(r2_curr, z_asph);
                // std::cout << dr::grad_enabled(r2_curr) << ", " << dr::grad_enabled(z_asph) << std::endl;

                Point3f plane_p(p_curr.x(), p_curr.y(), z_asph);
                err = dr::abs(p_curr.z() - plane_p.z());
                Normal3f plane_n = normal(p_curr);


                // intersect ray with tangent plane
                t = dr::dot(plane_n, plane_p - ray.o) / dr::dot(plane_n, ray.d);
                p_curr = ray(t);
                r2_curr = dr::sqr(p_curr.x()) + dr::sqr(p_curr.y());

                itr++;
                active &= (err > TOL) && (itr < 10);
            }

            // check whether newton converged
            active = err < TOL;

            // TODO: if newton fails, exit? or retry with bisection?
            if (dr::none_or<false>(active)) {
                return si;
            }

            active &= t > 0.0f;

            Point3f p_surface = ray(t);
            si.t = t;
            si.p = p_surface;
            si.n = normal(p_surface);

            return si;
        }

        Normal3f normal(const Point3f &p) const override {
            Vector2f radial(p.x() * dr::rcp(LensInterface<Float, Spectrum>::m_element_radius), 
                            p.y() * dr::rcp(LensInterface<Float, Spectrum>::m_element_radius));
            Float r2_ = dr::squared_norm(radial);
            radial = dr::select(r2_ >= 4.f * dr::Epsilon<Float>, 
                                radial * dr::rsqrt(r2_), Vector2f(0.0f));
            Float z_grad = _eval_asph_grad(r2_);
            Normal3f normal(z_grad * radial.x(), z_grad * radial.y(), -1.0f);
            return dr::normalize(normal);
        }

        void offset_along_axis(Float delta) override {
            LensInterface<Float, Spectrum>::m_z_intercept += delta;
        }

        std::string to_string() const override {
            using string::indent;
            std::ostringstream oss;

            oss << "AsphericalLens[" << std::endl
                << "  z_intercept = " << LensInterface<Float, Spectrum>::m_z_intercept << "," << std::endl
                << "  curvature = " << m_c << "," << std::endl
                << "  kappa = " << m_K << "," << std::endl
                << "]";
            return oss.str();
        }
    private:
        Float m_c;
        Float m_K;
        Float m_r_scale;
        std::vector<Float> m_ai;    // TODO: switch to fixed-size array?

        inline Float _eval_conic(Float r2_) const {
            Float sqr_term = 1.f - (1.f + m_K) * dr::sqr(m_c) * r2_;
            Float z_ = m_c * r2_ * dr::rcp(1.f + dr::sqrt(sqr_term));
            return z_;
        }

        inline Float _eval_conic_grad(Float r_) const {
            // Float r_ = r * dr::rcp(LensInterface<Float, Spectrum>::m_element_radius);
            Float cr = m_c * r_;
            Float sqr_term = 1.f - (1.f + m_K) * dr::sqr(cr);
            Float dz_ = cr * dr::rsqrt(sqr_term);
            return dz_;
        }

        // Evaluate the asphere polynomial using Horner's method
        Float eval_asph(Float r2) const {
            Float r2_ = r2 * dr::rcp(dr::sqr(LensInterface<Float, Spectrum>::m_element_radius));
            Float z_ = 0.f;
            for (int i = m_ai.size() - 1; i >= 0; --i) {
                z_ = dr::fmadd(z_, r2_, m_ai.at(i));
            }
            z_ *= dr::sqr(r2_);
            z_ += _eval_conic(r2_);
            return -z_ * LensInterface<Float, Spectrum>::m_element_radius + LensInterface<Float, Spectrum>::m_z_intercept;
        }

        Float eval_asph_grad(Float r2) const {
            Float r2_ = r2 * dr::rcp(dr::sqr(LensInterface<Float, Spectrum>::m_element_radius));
            Float r_ = dr::sqrt(r2_);
            Float z_ = 0.f;
            for (int i = m_ai.size() - 1; i >= 0; --i) {
                z_ = dr::fmadd(z_, r2_, (2.f * i + 4.f) * m_ai.at(i));
            }
            z_ *= r2_ * r_;
            z_ += _eval_conic_grad(r_);
            return -z_;
        }

        // unitless version of eval_asph_grad; the input r2_ is unitless
        Float _eval_asph_grad(Float r2_) const {
            Float r_ = dr::sqrt(r2_);
            Float z_ = 0.f;
            for (int i = m_ai.size() - 1; i >= 0; --i) {
                z_ = dr::fmadd(z_, r2_, (2.f * i + 4.f) * m_ai.at(i));
            }
            z_ *= r2_ * r_;
            z_ += _eval_conic_grad(r_);
            return -z_;
        }



        Float _test_asph_grad_fd(Float r, float delta=0.0001f) const {
            Float f = eval_asph(dr::sqr(r));
            Float r_dr = r + delta * r;
            Float f_dr = eval_asph(dr::sqr(r_dr));
            Float grad_fd = (f_dr - f) * dr::rcp(delta * r);
            Float grad_an = eval_asph_grad(dr::sqr(r));
            return dr::abs(grad_fd - grad_an);
        }

        void _test_asph_grad() const {
            Float r = 0.1f * LensInterface<Float, Spectrum>::m_element_radius;

            std::cout << "FD test begin.\n";

            // logspace(-8, 0, 50)
            std::vector<float> eps_list = {
                1.00000000e-8, 1.45634848e-8, 2.12095089e-8, 3.08884360e-8,
                4.49843267e-8, 6.55128557e-8, 9.54095476e-8, 1.38949549e-7,
                2.02358965e-7, 2.94705170e-7, 4.29193426e-7, 6.25055193e-7,
                9.10298178e-7, 1.32571137e-6, 1.93069773e-6, 2.81176870e-6,
                4.09491506e-6, 5.96362332e-6, 8.68511374e-6, 1.26485522e-5,
                1.84206997e-5, 2.68269580e-5, 3.90693994e-5, 5.68986603e-5,
                8.28642773e-5, 1.20679264e-4, 1.75751062e-4, 2.55954792e-4,
                3.72759372e-4, 5.42867544e-4, 7.90604321e-4, 1.15139540e-3,
                1.67683294e-3, 2.44205309e-3, 3.55648031e-3, 5.17947468e-3,
                7.54312006e-3, 1.09854114e-2, 1.59985872e-2, 2.32995181e-2,
                3.39322177e-2, 4.94171336e-2, 7.19685673e-2, 1.04811313e-1,
                1.52641797e-1, 2.22299648e-1, 3.23745754e-1, 4.71486636e-1,
                6.86648845e-1, 1.00000000e+0};

            for (size_t i = 0; i < eps_list.size(); ++i) {
                std::cout << _test_asph_grad_fd(r, eps_list.at(i)) << ", ";
            }
            
            std::cout << "\nFD test complete." << std::endl;
        }


        std::tuple<Float, Mask> intersect_conic(const Ray3f& ray) const {
            Vector3f o = ray.o - Vector3f(0.f, 0.f, LensInterface<Float, Spectrum>::m_z_intercept),
                     d = ray.d;

            Float R = LensInterface<Float, Spectrum>::m_element_radius;
            o *= dr::rcp(R);

            Float A = m_c * (1.f + m_K * dr::sqr(d.z())),
                  B = 2.f * (m_c * (dr::dot(o, d) +  m_K * o.z() * d.z()) - d.z()),
                  C = m_c * (dr::squared_norm(o) + m_K * dr::sqr(o.z())) - 2.f * o.z();
            
            auto [valid, t0, t1] = math::solve_quadratic(A, B, C);

            // if valid == false, discriminant is negative and ray misses the real+virtual 
            // surfaces completely. return invalid
            if (dr::none(valid)) {
                return {t0, valid};
            }

            t0 *= R;
            t1 *= R;

            // test each root to see whether it lies on the real surface
            Float t0_ztest = m_c * (ray.o.z() + t0 * ray.d.z() - LensInterface<Float, Spectrum>::m_z_intercept),
                  t1_ztest = m_c * (ray.o.z() + t1 * ray.d.z() - LensInterface<Float, Spectrum>::m_z_intercept);

            Mask t0_valid = valid & (t0_ztest > 0.0f),
                 t1_valid = valid & (t1_ztest > 0.0f);

            t0_valid &= t0_ztest <= dr::select(m_K > -1.0f, R * dr::rcp(1.0f + m_K), dr::Infinity<Float>);
            t1_valid &= t1_ztest <= dr::select(m_K > -1.0f, R * dr::rcp(1.0f + m_K), dr::Infinity<Float>);

            // to have a valid solution, at least one root must be on the real surface
            valid &= (t0_valid | t1_valid);

            // if neither root is valid, both lie on the virtual surface.
            // reject them and return invalid
            if (dr::none(valid)) {
                return {t0, valid};
            }

            // if both roots are on the valid surface, take the closest non-negative one
            if (dr::any(t0_valid & t1_valid)) {
                return {dr::select(t0 > 0.0f, t0, t1), valid};
            }
        
            // if only one root is valid, pick the valid one
            return {dr::select(t0_valid, t0, t1), valid};
        }
};

enum class InterfaceType { sphere, asphere, aperture, plane };


template <typename Float, typename Spectrum>
class RealisticLensCamera final : public ProjectiveCamera<Float, Spectrum> {
public:
    MI_IMPORT_BASE(ProjectiveCamera, m_to_world, m_needs_sample_3, m_film, m_sampler,
                    m_resolution, m_shutter_open, m_shutter_open_time, m_near_clip,
                    m_far_clip, m_focus_distance, sample_wavelengths)
    MI_IMPORT_TYPES()

    RealisticLensCamera(const Properties &props) : Base(props) {
        if (m_to_world.scalar().has_scale())
            Throw("Scale factors in the camera-to-world transformation are not allowed!");

        update_camera_transforms();

        m_film_z_position = 0.0f;
        m_film_diagonal = dr::norm(m_film->get_physical_size());
        m_needs_sample_3 = true;

        run_tests();
        
        std::string lens_type = props.get<std::string>("lens_design", "singlet");

        Float object_distance, focal_length, lens_diameter;
        object_distance = props.get<Float>("object_distance", 6.0f);
        focal_length    = props.get<Float>("lens_focal_length",    0.05f);
        lens_diameter   = props.get<Float>("lens_diameter",   0.01f);
        m_stopdown_ratio = props.get<Float>("stop_ratio", 1.0f);
        m_sample_exit_pupil = props.get<bool>("sample_exit_pupil", false);

        // build lens materials and geometry
        if (lens_type == "singlet") {
            build_singlet_lens(object_distance, focal_length, lens_diameter / 2);
        } else if (lens_type == "doublet") {
            build_doublet_lens(object_distance, focal_length / 2, lens_diameter / 2);
        } else if (lens_type == "tessar") {
            build_tessar_lens();
        } else if (lens_type == "helios") {
            build_helios_lens();
        } else if (lens_type == "jupiter") {
            build_jupiter_lens();
        } else if (lens_type == "fisheye") {
            build_fisheye_lens();
        } else if (lens_type == "gauss") {
            build_double_gauss_smith();
        } else if (lens_type == "hypercentric") {
            // aperture radius in mm
            Float aperture_size = props.get<Float>("ap_size", 1.0f); 
            // aperture<->lens distance in units of focal lengths.
            // telecentric optics: z = 1.0
            // hypercentric optics: 1.0 < z <= 2.0
            Float aperture_offset = props.get<Float>("ap_offset", 2.0f); 
            build_hypercentric_lens(aperture_size, aperture_offset);
        } else if (lens_type == "asph") {
            build_asph_lens();
        } else if (lens_type == "exp1a") {
            build_exp1_doublet(1.5689525390422485f, 0.0263051608728981f);
        } else if (lens_type == "exp1b") {
            build_exp1_doublet(1.507009332993809f, 0.04781645616233479f);
        } else if (lens_type == "exp1c") {
            build_exp1_doublet(1.5030211256389316f, 0.04920145883751409f);
        } else if (lens_type == "exp2a") {
            build_exp2_nikon(0);
        } else if (lens_type == "exp2b") {
            build_exp2_nikon(1);
        } else if (lens_type == "exp2c") {
            build_exp2_nikon(2);
        } else {
            build_singlet_lens(object_distance, focal_length, lens_diameter / 2);
        }
        // finish initializing lens
        
        // Step 2: fine-focusing

        Float delta = focus_thick_lens(object_distance);

        for (const auto &interface : m_interfaces) {
            interface->offset_along_axis(dr::select(dr::isnan(delta), 0.0f, -delta));
        }

        std::cout << "Fine focus adjustment: " << delta << std::endl;
        m_rear_element_z = m_interfaces.front()->get_z();
        m_rear_element_radius = m_interfaces.front()->get_radius();
        m_lens_terminal_z = m_interfaces.back()->get_z() + dr::abs(m_interfaces.back()->get_radius());

        // Step 3: build exit pupil LUT
        m_qmc_sampler = new RadicalInverse();

        if (m_sample_exit_pupil) {
            std::cout << "Computing exit pupil LUT...\n";
            compute_exit_pupil_bounds();
            std::cout << "LUT complete!\n";
        }

        if (TEST_EXIT_PUPIL) {
            Float r_pupil = props.get<Float>("pupil_render_pos", 0.f);
            Float wv = props.get<Float>("pupil_render_wv", 589.3f);
            BoundingBox2f pupil_bound = bound_exit_pupil(r_pupil, r_pupil);
            std::cout << "BBOX, " << pupil_bound.min.x() << ", " << pupil_bound.min.y() << ", "
                                << pupil_bound.max.x() << ", " << pupil_bound.max.y() << "\n";
            render_exit_pupil(r_pupil, wv, 1 << 20);
        }

        // draw_cross_section(32);
    }


    void build_singlet_lens(Float object_distance, Float curvature_radius, Float lens_radius) {
        // place the film plane at the image formation distance `xi` away from the lens
        // equivalently, keep the film plane at z=0 and move the lens to `z_intercept` = `xi`

        // clamp to ensure a real image is formed
        Float distance = dr::maximum(object_distance, 4.001f * curvature_radius);

        // set the lens position using the *thin lens* equation; use this to validate
        // that `focus_thick_lens()` is behaving correctly. 
        Float z_intercept = 0.5f * distance * (1.f - dr::sqrt(1.f - 4.f * curvature_radius / distance));
        Float thickness = 2.f * curvature_radius * (1.f - dr::sqrt(1.f - (lens_radius / curvature_radius) * (lens_radius / curvature_radius)));

        DispersiveMaterial<Float, Spectrum> air_material = DispersiveMaterial<Float, Spectrum>("Air", 1.000277f, 0.0f);
        DispersiveMaterial<Float, Spectrum> glass_material = DispersiveMaterial<Float, Spectrum>("NBK7", 1.5046f, 5*0.00420f);
        auto lens1 = new SpheroidLens<Float, Spectrum>(curvature_radius, lens_radius, z_intercept, air_material, glass_material);
        m_interfaces.push_back(lens1);
        auto lens2 = new SpheroidLens<Float, Spectrum>(-curvature_radius, lens_radius, z_intercept + thickness, glass_material, air_material);
        m_interfaces.push_back(lens2);
    }


    void build_doublet_lens(Float object_distance, Float R, Float lens_radius) {
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
    }

    void build_tessar_lens() {
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

        std::vector<InterfaceType> interfaceTypes = {
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::aperture,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
        };

        std::vector<float> curv_radii  = {35.034f, -296.111f, -63.028f, 31.297f, 100000000, -86.620f, 45.344f, -43.567f};
        std::vector<float> thicknesses = {3.567f, 2.286f, 2.290f, 2.289f, 1.999f, 2.286f, 9.941f, 86.917f};
        std::vector<float> elem_radii  = {9.0f, 9.0f, 6.0f, 6.0f, 4.65f, 8.0f, 8.0f, 8.0f};
        std::vector<float> kappas = {};
        std::vector<std::vector<Float>> ai_list = {};
        for (size_t i = 0; i < thicknesses.size(); ++i) {
            kappas.push_back(0.0f);
            ai_list.push_back({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        }

        std::vector<DispersiveMaterial<Float, Spectrum>> mats = {
            air,
            NLAK9,
            air,
            F2,
            air,
            air,
            K10,
            NLAK9,
            air,
        };

        build_lens_from_data(interfaceTypes, thicknesses, curv_radii, elem_radii, kappas, ai_list, mats);
    }


    void build_helios_lens() {
        // Helios 44M-4 (swirly bokeh)
        // https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/GOI/ST01FB06.txt


        DispersiveMaterial<Float, Spectrum> air = 
            DispersiveMaterial<Float, Spectrum>("Air", 1.000277f, 0.0f);
        DispersiveMaterial<Float, Spectrum> BF16_glass = 
            DispersiveMaterial<Float, Spectrum>("BF16_glass", 1.648275034f, 0.007806736f);
        DispersiveMaterial<Float, Spectrum> TK14_glass = 
            DispersiveMaterial<Float, Spectrum>("TK14_glass", 1.597547619f, 0.005351918f);
        DispersiveMaterial<Float, Spectrum> LF7_glass = 
            DispersiveMaterial<Float, Spectrum>("LF7_glass",  1.551328271f, 0.008025103f);

        std::vector<InterfaceType> interfaceTypes = {
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::aperture,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
        };

        std::vector<float> curv_radii  = {38.07f, 136.365f, 25.33f, -124.225f, 15.995f, 1.00E+08, -16.62f, 66.085f, -22.21f, 191.54f, -52.725f};
        std::vector<float> thicknesses = {4.81f, 2.26f, 9.07f, 1.31f, 4.7f, 4.63f, 1.32f, 6.25f, 0.5f, 4.94f, 38.08f};
        std::vector<float> elem_radii  = {14.75f, 14.75f, 13.2f, 11.6f, 9.75f, 9.575f, 9.35f, 10.2f, 10.6f, 12.35f, 12.35f};
        std::vector<float> kappas = {};
        std::vector<std::vector<Float>> ai_list = {};
        for (size_t i = 0; i < thicknesses.size(); ++i) {
            kappas.push_back(0.0f);
            ai_list.push_back({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        }

        std::vector<DispersiveMaterial<Float, Spectrum>> mats = {
            air,
            TK14_glass,
            air,
            TK14_glass,
            LF7_glass,
            air,
            air,
            LF7_glass,
            TK14_glass,
            air,
            BF16_glass,
            air,
        };

        build_lens_from_data(interfaceTypes, thicknesses, curv_radii, elem_radii, kappas, ai_list, mats);
    }


    void build_jupiter_lens() {
        // Jupiter-9 (swirly bokeh)
        // https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/GOI/ST01FB43.txt


        DispersiveMaterial<Float, Spectrum> air = 
            DispersiveMaterial<Float, Spectrum>("Air", 1.000277f, 0.0f);
        DispersiveMaterial<Float, Spectrum> TK16_glass = 
            DispersiveMaterial<Float, Spectrum>("TK16_glass", 1.596466676f, 0.00558386f);
        DispersiveMaterial<Float, Spectrum> BF13_glass = 
            DispersiveMaterial<Float, Spectrum>("BF13_glass", 1.618336084f, 0.007302944f);
        DispersiveMaterial<Float, Spectrum> K1_glass = 
            DispersiveMaterial<Float, Spectrum>("K1_glass",  1.486688667f, 0.00398663f);
        DispersiveMaterial<Float, Spectrum> TF2_glass = 
            DispersiveMaterial<Float, Spectrum>("TF2_glass", 1.637217608f, 0.012112489f);
        DispersiveMaterial<Float, Spectrum> OF1_glass = 
            DispersiveMaterial<Float, Spectrum>("OF1_glass", 1.513488027f, 0.005500433f);
        DispersiveMaterial<Float, Spectrum> BF7_glass = 
            DispersiveMaterial<Float, Spectrum>("BF7_glass", 1.562693323f, 0.005811246f);

        std::vector<InterfaceType> interfaceTypes = {
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::aperture,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
        };

        std::vector<float> curv_radii  = {46.45f, 268.5f, 25.94f, 52.0f, -264.2f, 16.444f, 1.00E+08, -1043.65f, 44.51f, -15.031f, -95.06f};
        std::vector<float> thicknesses = {5.6f, 0.4f, 5.8f, 7.5f, 1.8f, 10.0f, 3.8f, 2.8f, 10.6f, 2.9f, 40.53f};
        std::vector<float> elem_radii  = {22.0f, 22.0f, 19.015f, 19.015f, 19.015f, 12.68f, 12.275f, 24.57f * 0.5f, 13.5f, 13.5f, 15.15f};
        std::vector<float> kappas = {};
        std::vector<std::vector<Float>> ai_list = {};
        for (size_t i = 0; i < thicknesses.size(); ++i) {
            kappas.push_back(0.0f);
            ai_list.push_back({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        }

        std::vector<DispersiveMaterial<Float, Spectrum>> mats = {
            air,
            TK16_glass,
            air,
            BF13_glass,
            K1_glass,
            TF2_glass,
            air,
            air,
            OF1_glass,
            BF13_glass,
            BF7_glass,
            air,
        };

        build_lens_from_data(interfaceTypes, thicknesses, curv_radii, elem_radii, kappas, ai_list, mats);
    }


    void build_fisheye_lens() {
        // Canon EF15mm f/2.8 Fisheye
        // https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/Data/JP1988-017421_Example03P.txt

        DispersiveMaterial<Float, Spectrum> air = 
            DispersiveMaterial<Float, Spectrum>("Air", 1.000277f, 0.0f);
        DispersiveMaterial<Float, Spectrum> glass_A = 
            DispersiveMaterial<Float, Spectrum>("glass_A", 1.5881276381075704f, 0.005202992085188941f);
        DispersiveMaterial<Float, Spectrum> glass_B = 
            DispersiveMaterial<Float, Spectrum>("glass_B", 1.793242496642434f, 0.018550536235572006f);
        DispersiveMaterial<Float, Spectrum> glass_C = 
            DispersiveMaterial<Float, Spectrum>("glass_C",  1.4770186893501427f, 0.003636419065560783f);
        DispersiveMaterial<Float, Spectrum> glass_D = 
            DispersiveMaterial<Float, Spectrum>("glass_D", 1.6021851259042148f, 0.005144827846028017f);
        DispersiveMaterial<Float, Spectrum> glass_E = 
            DispersiveMaterial<Float, Spectrum>("glass_E", 1.4983808648479255f, 0.004423976662977713f);

        // int num_elements = 16;
        // int aperture_index = 8;
        std::vector<InterfaceType> interfaceTypes = {
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::aperture,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
        };
        std::vector<float> curv_radii  = {78.06f, 15.9f, 22.22f, 13.27f, 127.88f, 22.35f, 32.04f, -190.22f, -1000000.0f, -289.77f, -29.1f, -100.42f, 29.39f, -25.73f, 43.88f, -43.88f};
        std::vector<float> thicknesses = {2.5f, 11.83f, 2.5f, 7.54f, 5.34f, 1.85f, 6.71f, 3.84f, 3.53f, 2.72f, 0.15f, 3.99f, 5.14f, 0.15f, 4.84f, 39.67f};
        std::vector<float> elem_radii  = {31.725f, 15.9f, 13.89f, 10.69f, 9.955f, 7.61f, 6.73f, 6.73f, 6.659f, 7.21f, 7.21f, 9.52f, 9.52f, 9.52f, 11.71f, 11.71f};
        std::vector<float> kappas = {};
        std::vector<std::vector<Float>> ai_list = {};
        for (size_t i = 0; i < thicknesses.size(); ++i) {
            kappas.push_back(0.0f);
            ai_list.push_back({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        }

        std::vector<DispersiveMaterial<Float, Spectrum>> mats = {
            air,
            glass_A,
            air,
            glass_D,
            air,
            glass_A,
            air,
            glass_B,
            air,
            air,
            glass_E,
            air,
            glass_B,
            glass_C,
            air,
            glass_C,
            air,
        };

        build_lens_from_data(interfaceTypes, thicknesses, curv_radii, elem_radii, kappas, ai_list, mats);
    }


    void build_double_gauss_smith() {
        // Double Gauss design from Smith, Modern Optical Engineering.

        DispersiveMaterial<Float, Spectrum> air = 
            DispersiveMaterial<Float, Spectrum>("Air", 1.000277f, 0.0f);
        DispersiveMaterial<Float, Spectrum> glass_A = 
            DispersiveMaterial<Float, Spectrum>("glass_A", 1.64855004723031f, 0.00744902140861971f);
        DispersiveMaterial<Float, Spectrum> glass_B = 
            DispersiveMaterial<Float, Spectrum>("glass_B", 1.66398266226799f, 0.0121606281020403f);
        DispersiveMaterial<Float, Spectrum> glass_C = 
            DispersiveMaterial<Float, Spectrum>("glass_C", 1.57907201321296f, 0.00830957940819446f);
        DispersiveMaterial<Float, Spectrum> glass_D = 
            DispersiveMaterial<Float, Spectrum>("glass_D", 1.64068415393588f, 0.00601335161083744f);
        DispersiveMaterial<Float, Spectrum> glass_E = 
            DispersiveMaterial<Float, Spectrum>("glass_E", 1.69447574875623f, 0.00782209786331075f);

        // int num_elements = 11;
        // int aperture_index = 5;
        std::vector<InterfaceType> interfaceTypes = {
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::aperture,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
        };
        std::vector<float> curv_radii  = {58.95f, 169.66f, 38.55f, 81.54f, 25.5f, 10000000.0f, -28.99f, 81.54f, -40.77f, 874.13f, -79.46f};
        std::vector<float> thicknesses = {7.52f, 0.24f, 8.05f, 6.55f, 11.41f, 9.0f, 2.36f, 12.13f, 0.38f, 6.44f, 72.228f};
        std::vector<float> elem_radii  = {25.2f, 25.2f, 23.0f, 23.0f, 18.0f, 17.1f, 17.0f, 20.0f, 20.0f, 20.0f, 20.0f};
        std::vector<float> kappas = {};
        std::vector<std::vector<Float>> ai_list = {};
        for (size_t i = 0; i < thicknesses.size(); ++i) {
            kappas.push_back(0.0f);
            ai_list.push_back({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        }

        std::vector<DispersiveMaterial<Float, Spectrum>> mats = {
            air,
            glass_A,
            air,
            glass_A,
            glass_B,
            air,
            air,
            glass_C,
            glass_D,
            air,
            glass_E,
            air,
        };

        build_lens_from_data(interfaceTypes, thicknesses, curv_radii, elem_radii, kappas, ai_list, mats);
    }


    void build_exp1_doublet(float mat_cauchy_A, float mat_cauchy_B) {
        DispersiveMaterial<Float, Spectrum> air = 
            DispersiveMaterial<Float, Spectrum>("Air", 1.000277f, 0.0f);
        DispersiveMaterial<Float, Spectrum> glass_A = 
            DispersiveMaterial<Float, Spectrum>("glass_A", 1.4560798389592584, 0.021086562958141358);
        DispersiveMaterial<Float, Spectrum> glass_B = 
            DispersiveMaterial<Float, Spectrum>("glass_B", mat_cauchy_A, mat_cauchy_B);

        // int num_elements = 4;
        // int aperture_index = 0;
        std::vector<InterfaceType> interfaceTypes = {
            InterfaceType::aperture,
            InterfaceType::sphere,
            InterfaceType::sphere,
            InterfaceType::sphere,
        };

        std::vector<float> curv_radii  = {1000.0f, 24.00000000f, -24.00000000f, -168.01068267f};
        std::vector<float> thicknesses = {0.0f, 3.00000000f, 2.25291824f, 46.74708176f};
        std::vector<float> elem_radii  = {8.0f, 8.00000000f, 8.00000000f, 8.00000000f};
        std::vector<float> kappas      = {0.0f, 0.0f, 0.0f, 0.0f};
        std::vector<std::vector<Float>> ai_list = {
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 1
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 2
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 3
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 4
        };

        std::vector<DispersiveMaterial<Float, Spectrum>> mats = {
            air,
            air,
            glass_A,
            glass_B,
            air,
        };

        build_lens_from_data(interfaceTypes, thicknesses, curv_radii, elem_radii, kappas, ai_list, mats);
    }

    void build_hypercentric_lens(Float ap_radius = 1.0f, Float focal_lengths = 2.0f) {
        DispersiveMaterial<Float, Spectrum> air = DispersiveMaterial<Float, Spectrum>("Air", 1.000277f, 0.0f);
        DispersiveMaterial<Float, Spectrum> SF6 = DispersiveMaterial<Float, Spectrum>("SF6", 1.757435971209294f, 0.016580283248837778f);

        float target_img_size = 5.0f;      // desired image size

        // design parameters for the primary lens (mm)
        float lens_radius = 120.0f;
        float focal_length = 210.0f;
        float track_length = 450.0f;
        float curvature_radius = 310.505660861f;
        float lens_thickness = 50.0f;

        Float z_ap_to_lens = focal_length * focal_lengths;
        Float z_film_to_ap = z_ap_to_lens * (target_img_size - ap_radius) * dr::rcp(lens_radius + ap_radius);

        size_t num_elements = 3;
        int aperture_index = 2;
        std::vector<Float> curv_radii  = { curvature_radius, -curvature_radius, dr::Infinity<Float> };
        std::vector<Float> thicknesses = { lens_thickness, z_ap_to_lens - lens_thickness * 0.5f, z_film_to_ap };
        std::vector<Float> elem_radii  = { lens_radius, lens_radius, ap_radius };
        std::vector<DispersiveMaterial<Float, Spectrum>> mats = { air, SF6, air, air };

        Float z_pos = 0.0f;
        Float thickness, curv_radius, elem_radius;
        for (int i = num_elements - 1; i >= 0; i--) {
            elem_radius = elem_radii.at(i);
            thickness   = thicknesses.at(i);
            z_pos += thickness;
            if (i == aperture_index) {
                m_interfaces.push_back(new ApertureStop<Float, Spectrum>(
                    0.001f * elem_radius, 
                    0.001f * z_pos, 
                    air));
            } else {
                curv_radius = curv_radii.at(i);
                m_interfaces.push_back(new SpheroidLens<Float, Spectrum>(
                    -0.001f * curv_radius, 
                    0.001f * elem_radius, 
                    0.001f * z_pos, 
                    mats.at(i + 1), 
                    mats.at(i)));
            }
        }
        m_film_z_position = 0.001f * (track_length - (thicknesses.at(0) + thicknesses.at(1) + thicknesses.at(2)));

        Float shutter_time = dr::sqr(z_ap_to_lens * dr::rcp(ap_radius * (ap_radius + lens_radius)));
        std::cout << "Recommended shutter time: " << shutter_time * 8.0f << "\n";
    }


    void build_asph_lens() {
        // TODO: find some way to deal with the weird aperture (it's currently excluded)
        // Parameters from:
        // https://patents.google.com/patent/US8934179B2/en

        DispersiveMaterial<Float, Spectrum> air = DispersiveMaterial<Float, Spectrum>("Air", 1.000277f, 0.0f);
        DispersiveMaterial<Float, Spectrum> glass_A = DispersiveMaterial<Float, Spectrum>("glass_A", 1.5206352150873f, 0.004988523354517577f);
        DispersiveMaterial<Float, Spectrum> glass_B = DispersiveMaterial<Float, Spectrum>("glass_B", 1.5949533129576456f, 0.013907192818823239f);
        DispersiveMaterial<Float, Spectrum> glass_C = DispersiveMaterial<Float, Spectrum>("glass_C", 1.5048569450665132f, 0.004216973209068582f);

        // size_t num_elements = 13;
        // int aperture_index = 0;
        // std::vector<float> curv_radii  = { 1e8, 1.754f, -5.259f, 18.175f, 2.111f, 49.667f, 9.971f, 3.479f, 21.778f, 2.402f, 1.334f, 1e8, 1e8 };
        // std::vector<float> thicknesses = { -0.225f, 0.655f, 0.025f, 0.27f, 0.35f, 0.516f, 0.187f, 0.605f, 0.573f, 0.8f, 0.3f, 0.3f, 0.607f };
        // std::vector<float> elem_radii  = { 0.89f, 1.026181818f, 1.026181818f, 1.026181818f, 1.026181818f, 1.211636364f, 1.211636364f, 1.446545455f, 1.557818182f, 1.842181818f, 2.373818182f, 2.670545455f, 2.670545455f };
        // std::vector<float> kappas  = { 0.0f, -1.898E+00, -1.818E+00, 0.000E+00, -2.723E-01, 0.000E+00, 3.438E+00, -3.702E+01, -3.345E+04, -1.855E+01, -4.858E+00, 0.0f, 0.0f };
        size_t num_elements = 12;
        int aperture_index = -1;
        std::vector<float> curv_radii  = { 1.754f, -5.259f, 18.175f, 2.111f, 49.667f, 9.971f, 3.479f, 21.778f, 2.402f, 1.334f, 1e8, 1e8 };
        std::vector<float> thicknesses = { 0.655f, 0.025f, 0.27f, 0.35f, 0.516f, 0.187f, 0.605f, 0.573f, 0.8f, 0.3f, 0.3f, 0.607f };
        // std::vector<float> elem_radii  = { 1.026181818f, 1.026181818f, 1.026181818f, 1.026181818f, 1.211636364f, 1.211636364f, 1.446545455f, 1.557818182f, 1.842181818f, 2.373818182f, 2.670545455f, 2.670545455f };
        std::vector<float> elem_radii  = { 0.89f, 1.026181818f, 1.026181818f, 1.026181818f, 1.211636364f, 1.211636364f, 1.446545455f, 1.557818182f, 1.842181818f, 2.373818182f, 2.670545455f, 2.670545455f };
        std::vector<float> kappas  = { -1.898E+00, -1.818E+00, 0.000E+00, -2.723E-01, 0.000E+00, 3.438E+00, -3.702E+01, -3.345E+04, -1.855E+01, -4.858E+00, 0.0f, 0.0f };

        std::vector<std::vector<Float>> ai_list = {
            // { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // aperture
            { 3.822E-02, -2.809E-02,  4.970E-02, -5.149E-02,  4.628E-03,  4.215E-03, -3.450E-03 },
            { 1.288E-01, -1.343E-01,  1.978E-02,  3.399E-04, -6.173E-04, -5.735E-04,  8.520E-12 },
            { 4.814E-02,  6.037E-02, -1.838E-01,  1.217E-01, -1.665E-02, -5.234E-04,  2.394E-04 },
            { -8.944E-02,  2.532E-01, -3.068E-01,  2.175E-01, -5.539E-02,  3.281E-03, -6.552E-07 },
            { -1.060E-01,  5.779E-02,  1.251E-03, -3.017E-02,  6.065E-02, -1.536E-02, -2.048E-03 },
            { -1.142E-01, -2.103E-02,  7.808E-03,  2.283E-02,  5.590E-05, -1.053E-03, -1.525E-04 },
            { 5.323E-02, -7.412E-02, -1.800E-02,  1.682E-02,  4.538E-03, -2.738E-03, -1.886E-05 },
            { -3.596E-02, 9.066E-02, -1.026E-01,  4.108E-02, -5.778E-03, -5.187E-05, -6.175E-06 },
            { -1.503E-01, 4.478E-02, -7.829E-03, -1.119E-03, 2.461E-04, 0.000E+00, 0.000E+00 },
            { -9.165E-02, 4.113E-02, -1.389E-02, 2.647E-03, -2.445E-04, 3.564E-06, 6.120E-07 },
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // plano-lenses
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }    // plano-lenses
        };

        std::vector<DispersiveMaterial<Float, Spectrum>> mats = {
            // air,
            air,
            glass_A,
            air,
            glass_B,
            air,
            glass_A,
            air,
            glass_A,
            air,
            glass_A,
            air,
            glass_C,
            air,
        };

        std::cout << "Array sizes match: " 
            << (curv_radii.size() == num_elements) << ", "
            << (thicknesses.size() == num_elements) << ", "
            << (elem_radii.size() == num_elements) << ", "
            << (kappas.size() == num_elements) << ", "
            << (ai_list.size() == num_elements) << ", "
            << (mats.size() == num_elements + 1) << "\n";



        Float z_pos = 0.0f;
        Float thickness, curv_radius, elem_radius, kappa;
        std::vector<Float> Ai;

        for (int i = num_elements - 1; i >= 0; i--) {
            elem_radius = elem_radii.at(i);
            thickness   = thicknesses.at(i);
            z_pos += thickness;
            if (i == aperture_index) {
                m_interfaces.push_back(new ApertureStop<Float, Spectrum>(
                    0.001f * elem_radius, 
                    0.001f * z_pos, 
                    air));
            } else {
                curv_radius = curv_radii.at(i);
                kappa = kappas.at(i);
                Ai = ai_list.at(i);
                m_interfaces.push_back(new AsphericalLens<Float, Spectrum>(
                    curv_radius, 
                    kappa,
                    0.001f * elem_radius, 
                    0.001f * z_pos, 
                    Ai,
                    mats.at(i + 1), 
                    mats.at(i)));
            }
        }
    }


    void build_exp2_nikon(int mode) {
        DispersiveMaterial<Float, Spectrum> air = DispersiveMaterial<Float, Spectrum>("Air", 1.000277f, 0.0f);
        DispersiveMaterial<Float, Spectrum> glass_A = DispersiveMaterial<Float, Spectrum>("glass_A", 1.5046464980225065f, 0.004220601200948007f),
                                            glass_B = DispersiveMaterial<Float, Spectrum>("glass_B", 1.8645930283464647f, 0.013233579146425922f),
                                            glass_C = DispersiveMaterial<Float, Spectrum>("glass_C", 1.5725988459469284f, 0.007952986594191858f),
                                            glass_D = DispersiveMaterial<Float, Spectrum>("glass_D", 1.6639243971469755f, 0.012180862092226684f),
                                            glass_E = DispersiveMaterial<Float, Spectrum>("glass_E", 1.8718384392978837f, 0.013530356122431513f),
                                            glass_F = DispersiveMaterial<Float, Spectrum>("glass_F", 1.5797734345478602f, 0.0045237938738434446f),
                                            glass_G = DispersiveMaterial<Float, Spectrum>("glass_G", 1.689190440676433f,  0.010873011246216379f),
                                            glass_H = DispersiveMaterial<Float, Spectrum>("glass_H", 1.5673889795706786f, 0.008789871710972213f),
                                            glass_I = DispersiveMaterial<Float, Spectrum>("glass_I", 1.6738434507974307f, 0.006826218099482183f),
                                            glass_J = DispersiveMaterial<Float, Spectrum>("glass_J", 1.4770641526601842f, 0.00362410356265244f);

        float scale = 35.0f / 1.57f;

        // size_t num_elements = 22;
        // int aperture_index = 11;
        std::vector<InterfaceType> interfaceTypes = {
            InterfaceType::sphere,    // 1 
            InterfaceType::sphere,    // 2
            InterfaceType::sphere,    // 3
            InterfaceType::sphere,    // 4
            InterfaceType::sphere,    // 5
            InterfaceType::sphere,    // 6
            InterfaceType::sphere,    // 7
            InterfaceType::sphere,    // 8
            InterfaceType::sphere,    // 9
            InterfaceType::sphere,    // 10
            InterfaceType::sphere,    // 11
            InterfaceType::aperture,  // 12, aperture
            InterfaceType::sphere,    // 13
            InterfaceType::sphere,    // 14
            InterfaceType::sphere,    // 15
            InterfaceType::sphere,    // 16
            InterfaceType::asphere,   // 17, asph
            InterfaceType::asphere,   // 18, asph
            InterfaceType::sphere,    // 19,
            InterfaceType::sphere,    // 20,
            InterfaceType::plane,     // 21, plano
            InterfaceType::plane      // 22, plano
        };

        std::vector<float> curv_radii  = { 5.2674f, 0.9607f, 1.442f, 10.2799f, 1.2154f, -1.0987f, 2.9183f, -1.6693f, 1.6429f, -1.4116f, -2.5725f,   1e8,  -0.973f, -24.0801f, 2.3756f, -1.3055f, -7.3169f,  -2.2f, -1.5449f, -7.2572f,   1e8,    1e8   };
        std::vector<float> thicknesses = {  0.102f,  0.309f, 0.246f,   0.083f,  0.411f,   0.088f,  0.258f,   0.009f,  0.379f,   0.069f,   0.118f, 0.604f,  0.051f,    0.009f,  0.282f,   0.239f,   0.122f, 0.154f,   0.083f,    0.75f, 0.074f, 0.0425f };
        std::vector<float> elem_radii  = {  0.847f,  0.696f, 0.661f,   0.625f,  0.546f,   0.524f,  0.586f,   0.601f,  0.624f,   0.613f,   0.607f, 0.555f,  0.476f,     0.49f,  0.543f,   0.569f,   0.604f, 0.627f,   0.662f,   0.712f,   1.2f,    1.2f };
        std::vector<float> kappas      = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

        std::vector<std::vector<Float>> ai_list = {
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 1
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 2
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 3
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 4
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 5
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 6
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 7
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 8
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 9
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 10
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 11
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 12, aperture
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 13
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 14
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 15
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 16
            { -0.24, -0.4268, 0.0, 0.0, 0.0, 0.0, 0.0 },   // 17, asph
            { -0.05053, -0.3491, 0.1459, 0.07718, 0.0, 0.0, 0.0 },   // 18, asph
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 19,
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 20,
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 21,
            { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },   // 22
        };

        if (mode == 1) {
            // zeroed-out aspheres
            std::cout << "Variant: no asphere\n";
            ai_list.at(16) = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            ai_list.at(17) = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        } else if (mode == 2) {
            // re-optimized aspheres
            std::cout << "Variant: re-optimized asphere\n";
            ai_list.at(16) = {-0.09490284323692322,-0.10803833603858948,-0.08069773763418198,-0.01646527275443077,-0.007415791507810354,-0.503877580165863,-3.2629644870758057};
            ai_list.at(17) = {0.07706142961978912, 0.06080370396375656, -0.015299857594072819, -0.16691075265407562, -0.38500601053237915, -0.5226516723632812, 0.1124117523431778};
        } 


        std::vector<DispersiveMaterial<Float, Spectrum>> mats = {
            air,
            glass_A,
            air,
            glass_B,
            glass_C,
            air,
            glass_D,
            glass_E,
            air,
            glass_F,
            glass_G,
            air,
            air,
            glass_H,
            air,
            glass_F,
            air,
            glass_I,
            air,
            glass_J,
            air,
            glass_A,
            air,
        };

        m_stopdown_ratio = 0.5f;

        build_lens_from_data(interfaceTypes, thicknesses, curv_radii, elem_radii, kappas, ai_list, mats, scale);
    }

    void build_lens_from_data(  std::vector<InterfaceType> interfaceTypes, 
                                std::vector<float> thicknesses,
                                std::vector<float> curv_radii,
                                std::vector<float> elem_radii,
                                std::vector<float> kappas,
                                std::vector<std::vector<Float>> ai_list,
                                std::vector<DispersiveMaterial<Float, Spectrum>> mats, 
                                float scale = 1.0f) {

        size_t num_elements = thicknesses.size();
        std::cout << "Array sizes match: " 
            << (curv_radii.size() == num_elements) << ", "
            << (elem_radii.size() == num_elements) << ", "
            << (ai_list.size() == num_elements) << ", "
            << (mats.size() == num_elements + 1) << "\n";

        Float z_pos = 0.0f;
        Float thickness, curv_radius, elem_radius, kappa;
        std::vector<Float> asph_coefficients;

        for (int i = num_elements - 1; i >= 0; i--) {
            elem_radius = elem_radii.at(i);
            thickness   = thicknesses.at(i);
            z_pos += thickness;

            switch (interfaceTypes[i]) {
                case InterfaceType::aperture: {
                    m_interfaces.push_back(new ApertureStop<Float, Spectrum>(
                        0.001f * scale * elem_radius * m_stopdown_ratio,
                        0.001f * scale * z_pos, 
                        mats.at(i + 1)          // air
                        ));
                    break;
                }
                case InterfaceType::plane: {
                    m_interfaces.push_back(new PlanoLens<Float, Spectrum>(
                        0.001f * scale * elem_radius, 
                        0.001f * scale * z_pos, 
                        mats.at(i + 1), 
                        mats.at(i)));
                    break;
                }
                case InterfaceType::asphere: {
                    curv_radius = curv_radii.at(i);
                    kappa = kappas.at(i);
                    asph_coefficients = ai_list.at(i);

                    for (size_t j = 0; j < asph_coefficients.size(); ++j) {
                        asph_coefficients.at(j) = asph_coefficients.at(j) / dr::pow(scale, 2*j + 3);
                    }

                    m_interfaces.push_back(new AsphericalLens<Float, Spectrum>(
                        scale * curv_radius, 
                        kappa,
                        0.001f * scale * elem_radius, 
                        0.001f * scale * z_pos, 
                        asph_coefficients,
                        mats.at(i + 1), 
                        mats.at(i)));
                    break;
                }
                case InterfaceType::sphere: {
                    curv_radius = curv_radii.at(i);
                    m_interfaces.push_back(new SpheroidLens<Float, Spectrum>(
                        -0.001f * scale * curv_radius, 
                        0.001f * scale * elem_radius, 
                        0.001f * scale * z_pos, 
                        mats.at(i + 1), 
                        mats.at(i)));
                    break;
                }
                default: {
                    curv_radius = curv_radii.at(i);
                    m_interfaces.push_back(new SpheroidLens<Float, Spectrum>(
                        -0.001f * scale * curv_radius, 
                        0.001f * scale * elem_radius, 
                        0.001f * scale * z_pos, 
                        mats.at(i + 1), 
                        mats.at(i)));
                    break;
                }
            }
        }
    }


    std::tuple<Ray3f, Mask> trace_ray_from_film(const Ray3f &ray) const {
        Mask active = true;
        Ray3f curr_ray(ray);

        for (const auto &interface : m_interfaces) {
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

    BoundingBox2f bound_exit_pupil(Float r0, Float r1) {
        size_t rays_per_segment = 1024 * 1024;
        Float rays_transmitted = 0;
        Point2f bbox_min = dr::Infinity<Point2f>,
                bbox_max = -dr::Infinity<Point2f>;
        Float rear_radius = m_interfaces.front()->get_radius() * 1.5f;
        Float rear_z = m_interfaces.front()->get_z();

        for (size_t i = 0; i < rays_per_segment; ++i) {
            Point3f p_film = Point3f(
                dr::lerp(r0, r1, (i + 0.5f) / rays_per_segment), 0.f, 0.f);
            Point3f p_rear(
                dr::lerp(-rear_radius, rear_radius, m_qmc_sampler->eval<Float>(0, i)),
                dr::lerp(-rear_radius, rear_radius, m_qmc_sampler->eval<Float>(1, i)),
                rear_z);
            
            Point2f p(p_rear.x(), p_rear.y());
            
            // calculation is active if point is outside the current bound area
            // Mask active = !pupil_bound.contains(Point2f(p_rear.x(), p_rear.y()));
            Mask active = !dr::all((p >= bbox_min) && (p <= bbox_max));

            // // LOGGING
            // std::cout << "\t ----- ray {"   << i << "} -----\n";
            // // std::cout << "\tBbox size, "    << pupil_bound << ",\n"
            // std::cout << "\tBbox size, "    << bbox_min << ", " << bbox_max << ",\n"
            //           << "\tCurr ray: o = " << ray.o << ", d = " << ray.d << ",\n"
            //           << "\t!contained: "   << active << ",\n";

            if (dr::none_or<false>(active)) {
                continue;
            }

            Wavelength wavelength;
            if constexpr (!is_spectral_v<Spectrum>) {
                // use nominal wavelength
                wavelength = Wavelength(589.3f);
            } else {
                // wavelength = dr::lerp(200.f, 700.f, (0.5f + (i & mask)) / num_wavelengths);
                wavelength = dr::lerp(380.f, 700.f, m_qmc_sampler->eval<Float>(2, i));
            }

            Ray3f ray(p_film, dr::normalize(Vector3f(p_rear - p_film)), 0.0f, wavelength);

            auto [ray_out, active_out] = trace_ray_from_film(ray);
            active &= active_out;

            // draw_ray_from_film(ray);

            // // LOGGING
            // std::cout << "\tRay transmitted: " << active_out << ",\n";
            // // only expand the pupil bbox if the ray was transmitted, 
            // // i.e. active_out == active == true
            // // std::cout << "\tBbox (before), " << pupil_bound.min << ", " << pupil_bound.max << ",\n";
            // std::cout << "\tBbox (before): " << bbox_min << ", " << bbox_max << ",\n";

            // bbox.expand(Point2f(p_rear.x(), p_rear.y()));
            Point2f new_min = dr::minimum(bbox_min, p),
                    new_max = dr::maximum(bbox_max, p);

            dr::masked(bbox_min, active) = new_min;
            dr::masked(bbox_max, active) = new_max;
            rays_transmitted += active;     // TODO: hsum?
            // dr::masked(rays_transmitted, active) = rays_transmitted + 1;

            // // LOGGING
            // // std::cout << "\tBbox (after), " << pupil_bound.min << ", " << pupil_bound.max << ",\n";
            // std::cout << "\tBbox (after): " << bbox_min << ", " << bbox_max << ",\n";
        }


        // handle zero transmission case
        dr::masked(bbox_min, rays_transmitted == 0) = Point2f(-rear_radius, -rear_radius);
        dr::masked(bbox_max, rays_transmitted == 0) = Point2f( rear_radius,  rear_radius);

        // expand by sample-sample spacing on the rear plane
        // TODO: i think there's an extra 2x factor that shouldn't be there?
        Float spacing = 4 * rear_radius * dr::sqrt(2.f / rays_per_segment);
        BoundingBox2f pupil_bound(bbox_min - spacing, bbox_max + spacing);

        return pupil_bound;
    }

    void compute_exit_pupil_bounds() {
        // differences from PBRT: we evaluate the exit pupil shape for the full
        // spectrum of visible wavelengths
        size_t num_segments = 64;
        m_exit_pupil_bounds.resize(num_segments);

        // TODO: workaround for bbox vector not working
        std::vector<Point2f> min_bounds = {}, max_bounds = {};

        for (size_t segment_id = 0; segment_id < num_segments; ++segment_id) {
            // TODO: initialization
            // BoundingBox2f pupil_bound();

            Float r0 = segment_id * m_film_diagonal / num_segments;
            Float r1 = (segment_id + 1) * m_film_diagonal / num_segments;

            // // LOGGING
            // std::cout << " ===== Segment {" << segment_id 
            //           << "}, r = [" << r0 
            //           << ", "       << r1 
            //           << "], z = "  << rear_z
            //           << " ===== \n";


            // initialize and launch rays
            // TODO: dr::arange?

            BoundingBox2f pupil_bound = bound_exit_pupil(r0, r1);
            m_exit_pupil_bounds[segment_id] = pupil_bound;

            // TODO: workaround for bbox vector not working
            min_bounds.push_back(pupil_bound.min);
            max_bounds.push_back(pupil_bound.max);

            // std::cout << "transmitted = " << rays_transmitted << " / " << rays_per_segment << ",\n";
            // std::cout << "Bbox = " << m_exit_pupil_bounds[segment_id] << ",\n";
        }

        // m_exit_pupil_bounds_ptr = dr::load<DynamicBuffer<Float>>(m_exit_pupil_bounds.data(), m_exit_pupil_bounds.size() * 4);
        m_min_bounds_ptr = dr::load<DynamicBuffer<Float>>(min_bounds.data(), min_bounds.size() * 2);
        m_max_bounds_ptr = dr::load<DynamicBuffer<Float>>(max_bounds.data(), max_bounds.size() * 2);
    }

    // sample a point on the rear plane using the exit pupil LUT
    Point3f sample_exit_pupil(const Point3f p_film, const Point2f aperture_sample, Float& bounds_area) const {
        Float r_film = dr::sqrt(dr::sqr(p_film.x()) + dr::sqr(p_film.y()));
        UInt32 r_idx = dr::floor2int<UInt32>(r_film / m_film_diagonal * m_exit_pupil_bounds.size());
        r_idx = dr::clamp(r_idx, 0, m_exit_pupil_bounds.size() - 1);

        // BoundingBox2f pupil_bounds = m_exit_pupil_bounds[r_idx];
        // bounds_area = pupil_bounds.volume();

        // TODO: workaround for bbox vector not working
        Point2f min_bound = Point2f(
            dr::gather<Float>(m_min_bounds_ptr, 2 * r_idx + 0),
            dr::gather<Float>(m_min_bounds_ptr, 2 * r_idx + 1));

        Point2f max_bound = Point2f(
            dr::gather<Float>(m_max_bounds_ptr, 2 * r_idx + 0),
            dr::gather<Float>(m_max_bounds_ptr, 2 * r_idx + 1));

        bounds_area = dr::prod(max_bound - min_bound);

        // Point2f p_pupil(
        //     dr::lerp(pupil_bounds.min.x(), pupil_bounds.max.x(), aperture_sample.x()),
        //     dr::lerp(pupil_bounds.min.y(), pupil_bounds.max.y(), aperture_sample.y()));
        
        // TODO: this is a workaround due to the bbox vector not working
        Point2f p_pupil(
            dr::lerp(min_bound.x(), max_bound.x(), aperture_sample.x()),
            dr::lerp(min_bound.y(), max_bound.y(), aperture_sample.y()));

        Float sin_theta = dr::select(r_film > 0.f, p_film.y() * dr::rcp(r_film), 0.f);
        Float cos_theta = dr::select(r_film > 0.f, p_film.x() * dr::rcp(r_film), 1.f);
        return Point3f(cos_theta * p_pupil.x() - sin_theta * p_pupil.y(),
                       sin_theta * p_pupil.x() + cos_theta * p_pupil.y(),
                       m_rear_element_z);
    }

    // render the exit pupil as seen from a radial position `r` on the film
    void render_exit_pupil(Float r, Float wavelength = 589.3f, size_t num_rays = 1 << 20) {
        Float rear_radius = m_interfaces.front()->get_radius() * 1.5f;
        Float rear_z = m_interfaces.front()->get_z();
        Float rays_transmitted = 0;

        for (size_t i = 0; i < num_rays; ++i) {
            Point3f p_film = Point3f(r, 0.f, 0.f);
            Point3f p_rear(
                dr::lerp(-rear_radius, rear_radius, m_qmc_sampler->eval<Float>(0, i)),
                dr::lerp(-rear_radius, rear_radius, m_qmc_sampler->eval<Float>(1, i)),
                rear_z);
            Point2f p(p_rear.x(), p_rear.y());

            Ray3f ray(p_film, dr::normalize(Vector3f(p_rear - p_film)), 0.0f, Wavelength(wavelength));
            auto [ray_out, active] = trace_ray_from_film(ray);

            std::cout << "POINT, " << p.x() << ", " << p.y() << ", " << active << "\n";
            rays_transmitted += active;
        }
    }

    // sample a point on the rear plane
    Point3f sample_rear_element(const Point3f /*p_film*/, const Point2f aperture_sample, Float& bounds_area) const {

        Point2f tmp = m_rear_element_radius * warp::square_to_uniform_disk_concentric(aperture_sample);
        Point3f p_rear(tmp.x(), tmp.y(), m_rear_element_z);

        bounds_area = dr::Pi<Float> * dr::sqr(m_rear_element_radius);

        return p_rear;
    }

    void draw_cross_section(int num_points) const {
        size_t vtx_idx = 0;
        std::vector<Point3f> points = {};
        std::vector<Normal3f> normals = {};
        std::vector<Point2i> edges = {};
        bool new_element;

        for (size_t surface_id = 0; surface_id < m_interfaces.size(); ++surface_id) {
            auto s = m_interfaces[surface_id];
            bool left_is_air =  string::to_lower(s->get_left_material()) == "air";
            bool right_is_air = string::to_lower(s->get_right_material()) == "air";

            std::vector<Point3f> p_list;
            std::vector<Normal3f> n_list;
            if (left_is_air) {
                // cases:
                //  1. air->air interface; aperture
                //  2. air->glass interface
                new_element = true;
                s->draw_surface(p_list, n_list, num_points, true);
            } else {
                // cases:
                //  3. glass->air interface
                //  4. glass->glass interface
                new_element = false;
                s->draw_surface(p_list, n_list, num_points, false);
            }

            // add points to the output
            for (size_t i = 0; i < p_list.size(); ++i) {
                points.push_back(p_list[i]);
                normals.push_back(n_list[i]);
                vtx_idx = points.size() - 1;
                if (new_element && i == 0) {
                    continue;
                }
                // connect curr point to previous point in the list
                edges.push_back(Point2i(vtx_idx - 1, vtx_idx));
            }

            // for glass-glass interface, draw the interface a second time
            if (!left_is_air && !right_is_air) {
                s->draw_surface(p_list, n_list, num_points, true);
                new_element = true;
                for (size_t i = 0; i < p_list.size(); ++i) {
                    points.push_back(p_list[i]);
                    normals.push_back(n_list[i]);
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

        std::cout << "\n\nedges = [";
        for (const auto& e : edges) {
            std::cout << e << ",\n";
        }
        std::cout << "]";

        std::cout << "\n\nnormals = [";
        for (const auto& n : normals) {
            std::cout << n << ",\n";
        }
        std::cout << "]";

    }


    // Traces a ray from the world side of the lens to the film side. The input
    // `ray` is assumed to be represented in camera space.
    std::tuple<Ray3f, Mask> trace_ray_from_world(const Ray3f &ray) const {
        Mask active = true;
        Ray3f curr_ray(ray);

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
        out_ray.o = out_ray((m_interfaces.front()->get_z() - out_ray.o.z()) * dr::rcp(out_ray.d.z()));

        // check that init_ray and backward(forward(init_ray)) are equal
        out_ray = out_ray.reverse();

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
        curr_ray.o = curr_ray((m_interfaces.front()->get_z() - curr_ray.o.z()) * dr::rcp(curr_ray.d.z()));

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
            Point3f(r, 0.f, m_interfaces.back()->get_z() + 1.f), 
            Vector3f(0.f, 0.f, -1.f),
            0.0f,
            Wavelength(589.3f));

        draw_ray_from_world(obj_ray);

        auto [obj_end_ray, active] = trace_ray_from_world(obj_ray);

        if (dr::none_or<false>(active)) {
            Throw("compute_thick_lens_approximation: world ray was not transmitted through lens!");
        }

        compute_cardinal_points(obj_ray, obj_end_ray, obj_plane, obj_focus);
        back_plane_z = obj_plane;
        back_focal_length = obj_plane - obj_focus;

        // image (film)-side quantities
        Float img_plane, img_focus;
        Ray3f img_ray = Ray3f(
            Point3f(r, 0.f, m_interfaces.front()->get_z() - 1.f), 
            Vector3f(0.f, 0.f, 1.f),
                        0.0f,
            Wavelength(589.3f));

        draw_ray_from_film(img_ray);

        auto [img_end_ray, active_] = trace_ray_from_film(img_ray);

        if (dr::none_or<false>(active_)) {
            Throw("compute_thick_lens_approximation: film ray was not transmitted through lens!");
        }

        compute_cardinal_points(img_ray, img_end_ray, img_plane, img_focus);
        front_plane_z = img_plane;
        front_focal_length = img_focus - img_plane;
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
        callback->put_parameter("focus_distance",  m_focus_distance,  +ParamFlags::NonDifferentiable);
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

        Point3f pmin(m_sample_to_film * Point3f(0.f, 0.f, 0.f)),
                pmax(m_sample_to_film * Point3f(1.f, 1.f, 0.f));

        m_image_rect.reset();
        m_image_rect.expand(Point2f(pmin.x(), pmin.y()));
        m_image_rect.expand(Point2f(pmax.x(), pmax.y()));
        m_normalization = 1.f / m_image_rect.volume();

        dr::make_opaque(m_film_to_sample, m_sample_to_film, 
                        m_image_rect, m_normalization);
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
        // Compute the sample position on the near plane. For RealisticCamera, this is 
        // the physical location of a point on the film, expressed in local camera space. 
        // The film occupies [-xmax, xmax] x [-ymax, ymax] x [0,0]. Meanwhile, 
        // `position_sample` is a uniform 2D sample distributed on [0,1]^2.
        Point3f film_p = m_sample_to_film *
                        Point3f(position_sample.x(), position_sample.y(), 0.f);

        // STAGE 2: APERTURE SAMPLING
        // Sample the exit pupil
        Float bounds_area(0.f);
        Point3f aperture_p;

        if (m_sample_exit_pupil) {
            aperture_p = sample_exit_pupil(film_p, aperture_sample, bounds_area);
        } else {
            aperture_p = sample_rear_element(film_p, aperture_sample, bounds_area);
        }

        // STAGE 3: RAY SETUP
        // ------------------------
        // Set up the film->pupil ray. The ray starts at `film_p` and is directed
        //  along the vector connecting `film_p` and `aperture_p`
        Vector3f d = dr::normalize(Vector3f(aperture_p - film_p));
        ray.o = film_p;
        ray.d = d;

        // Trace the ray through the lens
        auto [ray_out, active_out] = trace_ray_from_film(ray);
        Vector3f d_out_local(ray_out.d);
        active &= active_out;

        // Kill rays that don't get through the lens
        dr::masked(wav_weight, !active) = dr::zeros<Spectrum>();

        // draw_ray_from_film(ray);
        // std::cout << ", ";
        // draw_ray_from_world(ray_out.reverse());

        // Convert ray_out from camera to world space
        ray_out.o.z() += m_film_z_position;
        ray_out.o = m_to_world.value().transform_affine(ray_out.o);
        ray_out.d = m_to_world.value() * d_out_local;
        // ------------------------

        // STAGE 4: POST-PROCESS
        // handle z-clipping
        // NOTE: the direction `d` in inv_z should be in the camera frame, i.e. before `m_to_world` is applied
        Float inv_z = dr::rcp(d_out_local.z());
        Float near_t = m_near_clip * inv_z,
              far_t  = m_far_clip * inv_z;
        ray_out.o += ray_out.d * near_t;
        ray_out.maxt = far_t - near_t;

        // std::cout << "Reciprocity test: " << test_trace_ray_from_world(ray) << std::endl;

        bool pupil_weight = true;
        Float ct = d_out_local.z();
        Float cos4t = dr::sqr(dr::sqr(ct));

        if (m_sample_exit_pupil && pupil_weight) {
            wav_weight *= ProjectiveCamera<Float,Spectrum>::shutter_open_time() * 
                            bounds_area * dr::rcp(dr::sqr(m_rear_element_z));
        } else {
            // TODO: I think this needs to be normalized by the bounds area too
            wav_weight *= cos4t;
        }

        return { ray_out, wav_weight };
    }

    // This method is mitsuba's version of pbrt's Sample_Wi(), which in turn
    // is the sensor version of the emitters' Sample_Li(). Given some position p
    // in the world, it samples one of the possible directions between the sensor 
    // and p, and then evaluates the sensor's emitted importance along that direction.
    // 
    // to be specific, we must construct a path from p to a (fractional) pixel position 
    // on the sensor film
    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample,
                     Mask active) const override {
        // Transform the reference point into the local coordinate system (no change)
        Transform4f trafo = m_to_world.value();
        Point3f ref_p = trafo.inverse().transform_affine(it.p);

        // std::cout << "a\n";

        // Check if `it.p` is outside of the clip range (no change)
        DirectionSample3f ds = dr::zeros<DirectionSample3f>();
        ds.pdf = 0.f;
        active &= (ref_p.z() >= m_near_clip) && (ref_p.z() <= m_far_clip);
        if (dr::none_or<false>(active))
            return { ds, dr::zeros<Spectrum>() };


        // std::cout << "b\n";

        // Sample a position on the aperture (in local coordinates)
        // here, we define the aperture as a circular region on the "front plane": the
        // plane tangent to the outermost element surface, closest to the world.
        // we then sample a point on this circular region. A factor of 1.5x is added
        // to the circle radius to account for rays that might enter at oblique angles.
        Float front_radius = 1.0f * m_interfaces.back()->get_radius();
        Float front_z = m_interfaces.back()->get_z();
        Point2f tmp = warp::square_to_uniform_disk_concentric(sample) * front_radius;
        Point3f aperture_p(tmp.x(), tmp.y(), front_z);

        // Compute the normalized direction vector from the aperture position to the referent point
        // (no change)
        Vector3f dir_ap2ref = ref_p - aperture_p;
        Float dist_ref = dr::norm(dir_ap2ref);
        Float inv_dist_ref = dr::rcp(dist_ref);
        dir_ap2ref *= inv_dist_ref;

        // Compute the corresponding point on the film (expressed in UV coordinates)
        // In our case, we trace a ray backwards from the front, world-facing lens element to the rear
        // TODO: account for effect of dispersion on ray path

        // Point3f scr = m_camera_to_sample.transform_affine(
        //     aperture_p + local_d * (m_focus_distance * inv_ct));
        Ray3f world_ray(ref_p, -dir_ap2ref, 0.0f, Wavelength(589.3f));

        // std::cout << world_ray << "\n";

        auto [ray_out, valid] = trace_ray_from_world(world_ray);

        // std::cout << "c\n";

        // if ray doesn't reach the rear of the lens, exit
        if (dr::none_or<false>(valid)) {
            return { ds, dr::zeros<Spectrum>() };
        }

        Point3f scr = m_film_to_sample.transform_affine(
            ray_out(-ray_out.o.z() / ray_out.d.z())
        );

        // std::cout << "Point: " << scr << "\n";

        // Compute importance value
        // Mask valid = dr::all(scr >= 0.f) && dr::all(scr <= 1.f);
        valid &= dr::all(dr::head<2>(scr) >= 0.f) && dr::all(dr::head<2>(scr) <= 1.f);
        // Float ct     = Frame3f::cos_theta(local_d),
        Float ct_film = Frame3f::cos_theta(-ray_out.d),
              inv_ct  = dr::rcp(ct_film);
        // TODO: need to account for cos4 weight?
        Float value = dr::select(valid, m_normalization * dr::sqr(dr::sqr(inv_ct)) * dr::sqr(m_rear_element_z), 0.f);   // TODO: correct?? d^2?
        // Float value = dr::select(valid, m_normalization / dr::sqr(dr::sqr(Frame3f::cos_theta(dir_ap2ref))) * dr::sqr(m_rear_element_z), 0.f);   // TODO: correct?? d^2?

        if (dr::none_or<false>(valid))
            return { ds, dr::zeros<Spectrum>() };

        // std::cout << "d\n";

        // Populate DirectionSample
        ds.uv   = dr::head<2>(scr) * m_resolution;      // OK
        ds.p    = trafo.transform_affine(aperture_p);   // OK
        ds.d    = (ds.p - it.p) * inv_dist_ref;             // OK
        ds.dist = dist_ref;                                 // OK
        ds.n    = trafo * Vector3f(0.f, 0.f, 1.f);      // TODO: not sure, OK i think? (normal of the front plane)

        // compute sample PDF
        // Float aperture_pdf = dr::rcp(dr::Pi<Float> * dr::sqr(m_aperture_radius));
        Float aperture_pdf = dr::rcp(dr::Pi<Float> * dr::sqr(front_radius));

        Float ct_ref = Frame3f::cos_theta(dir_ap2ref);
        ds.pdf = dr::select(valid, aperture_pdf * dist_ref * dist_ref * dr::rcp(ct_ref), 0.f);     // convert pdf(ap<->ref) to solid angle (correct)
        // ds.pdf = dr::select(valid, aperture_pdf * dr::sqr(m_rear_element_z) * dr::rcp(ct_film), 0.f);   // wrong one

        return { ds, Spectrum(value * inv_dist_ref * inv_dist_ref * ct_ref) };   // TODO: sample *ref* geometry term, so include ct_ref
        // return { ds, Spectrum(value * dr::rcp(inv_ct * dr::sqr(m_rear_element_z))) };   // wrong one
    }

    // no change
    ScalarBoundingBox3f bbox() const override {
        ScalarPoint3f p = m_to_world.scalar() * ScalarPoint3f(0.f);
        return ScalarBoundingBox3f(p, p);
    }

    std::string to_string() const override {
        using string::indent;

        std::ostringstream oss;
        oss << "RealisticLensCamera[" << std::endl
            // << "  x_fov = " << m_x_fov << "," << std::endl
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
    BoundingBox2f m_image_rect;
    std::vector<BoundingBox2f> m_exit_pupil_bounds;
    // DynamicBuffer<Float> m_exit_pupil_bounds_ptr;
    DynamicBuffer<Float> m_min_bounds_ptr;
    DynamicBuffer<Float> m_max_bounds_ptr;
    Float m_film_diagonal;
    Float m_normalization;
    // TODO: replace pointer -> ref
    std::vector<LensInterface<Float, Spectrum>*> m_interfaces;
    Float m_rear_element_z, m_rear_element_radius, m_lens_terminal_z;
    ref<RadicalInverse> m_qmc_sampler;
    Float m_film_z_position;
    Float m_stopdown_ratio;
    bool m_sample_exit_pupil;




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
    }
};

MI_IMPLEMENT_CLASS_VARIANT(RealisticLensCamera, ProjectiveCamera)
MI_EXPORT_PLUGIN(RealisticLensCamera, "Realistic Lens Camera");
NAMESPACE_END(mitsuba)
