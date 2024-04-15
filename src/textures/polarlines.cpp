#include <mitsuba/render/texture.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _texture-checkerboard:

Polarlines texture (:monosp:`checkerlines`)
---------------------------------------------

.. pluginparameters::

 * - color0, color1
   - |spectrum| or |texture|
   - Color values for the two differently-colored patches (Default: 0.4 and 0.2)
   - |exposed|, |differentiable|

 * - to_uv
   - |transform|
   - Specifies an optional 3x3 UV transformation matrix. A 4x4 matrix can also be provided.
     In that case, the last row and columns will be ignored.  (Default: none)
   - |exposed|

This plugin provides a simple procedural outlined checkerboard texture with customizable colors.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/texture_checkerboard.jpg
   :caption: Polarlines applied to the material test object as well as the ground plane.
.. subfigend::
    :label: fig-texture-checkerboard

.. tabs::
    .. code-tab:: xml
        :name: checkerboard-texture

        <texture type="checkerboard">
            <rgb name="color0" value="0.1, 0.1, 0.1"/>
            <rgb name="color1" value="0.5, 0.5, 0.5"/>
        </texture>

    .. code-tab:: python

        'type': 'checkerboard',
        'color0': [0.1, 0.1, 0.1],
        'color1': [0.5, 0.5, 0.5]

 */

template <typename Float, typename Spectrum>
class Polarlines final : public Texture<Float, Spectrum> {
public:
    MI_IMPORT_TYPES(Texture)

    Polarlines(const Properties &props) : Texture(props) {
        m_color0 = props.texture<Texture>("color0", .4f);
        m_color1 = props.texture<Texture>("color1", .05f);
        m_linewidth = props.get<Float>("linewidth", .05f);
        m_transform = props.get<ScalarTransform3f>("to_uv", ScalarTransform3f());
        m_spokes = props.get<Int32>("spokes", 16);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("to_uv", m_transform,    +ParamFlags::NonDifferentiable);
        callback->put_object("color0",   m_color0.get(), +ParamFlags::Differentiable);
        callback->put_object("color1",   m_color1.get(), +ParamFlags::Differentiable);
        // callback->put_object("linewidth",   m_linewidth, +ParamFlags::NonDifferentiable);
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f &it, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        Point2f uv = m_transform.transform_affine(2.f * it.uv - 1.f);
        Float theta = dr::atan2(uv.x(), uv.y());
        Float radius = dr::norm(uv);
        // TODO: replace with inv_pi
        theta = theta / (2.f * dr::Pi<Float>) * m_spokes;

        dr::mask_t<Float> mask = 
            (dr::abs(2.f * (radius - dr::floor(radius)) - 1.f) > 1.f - m_linewidth) | 
            (dr::abs(2.f * (theta - dr::floor(theta)) - 1.f) > 1.f - m_linewidth * m_spokes / (2.f * dr::Pi<Float> * radius));

        UnpolarizedSpectrum result = dr::zeros<UnpolarizedSpectrum>();

        Mask m0 = mask,
             m1 = !m0;

        m0 &= active; m1 &= active;

        if (dr::any_or<true>(m0))
            result[m0] = m_color0->eval(it, m0);

        if (dr::any_or<true>(m1))
            result[m1] = m_color1->eval(it, m1);

        return result;
    }

    Float eval_1(const SurfaceInteraction3f &it, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        Point2f uv = m_transform.transform_affine(2.f * it.uv - 1.f);
        Float theta = dr::atan2(uv.x(), uv.y());
        Float radius = dr::norm(uv);
        // TODO: replace with inv_pi
        theta = theta / (2.f * dr::Pi<Float>) * m_spokes;

        dr::mask_t<Float> mask = 
            (dr::abs(2.f * (radius - dr::floor(radius)) - 1.f) > 1.f - m_linewidth) | 
            (dr::abs(2.f * (theta - dr::floor(theta)) - 1.f) > 1.f - m_linewidth * m_spokes / (2.f * dr::Pi<Float> * radius));

        Float result = 0.0f;

        Mask m0 = mask,
             m1 = !m0;

        m0 &= active; m1 &= active;

        if (dr::any_or<true>(m0))
            dr::masked(result, m0) = m_color0->eval_1(it, m0);

        if (dr::any_or<true>(m1))
            dr::masked(result, m1) = m_color1->eval_1(it, m1);

        return result;
    }

    // TODO: need to modify? what is this for?
    Float mean() const override {
        return .5f * (m_color0->mean() + m_color1->mean());
    }

    bool is_spatially_varying() const override { return true; }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Polarlines[" << std::endl
            << "  color0 = " << string::indent(m_color0) << std::endl
            << "  color1 = " << string::indent(m_color1) << std::endl
            << "  spokes = " << string::indent(m_spokes) << std::endl
            << "  linewidth = " << string::indent(m_linewidth) << std::endl
            << "  transform = " << string::indent(m_transform) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
protected:
    ref<Texture> m_color0;
    ref<Texture> m_color1;
    Float m_linewidth;
    Int32 m_spokes;
    ScalarTransform3f m_transform;
};

MI_IMPLEMENT_CLASS_VARIANT(Polarlines, Texture)
MI_EXPORT_PLUGIN(Polarlines, "Polarlines texture")
NAMESPACE_END(mitsuba)
