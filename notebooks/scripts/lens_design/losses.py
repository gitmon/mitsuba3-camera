import sys
sys.path.insert(0, '/home/chuah/mitsuba3-camera/build/python')
sys.path.append("..")
import drjit as dr
import mitsuba as mi


def rms_loss(image: mi.Color3f):
    '''
    Compute the loss as the root-mean-squared (RMS) radius of the projected spot in the rendered image.
    Note that this loss should only be used when the image contains a single spot! If trying to
    optimize multiple spots concurrently (e.g. spot sizes from several field angles), a new sensor
    should be used for each spot

    Input: 
        - image: mi.Color3f. The rendered image of the spot on the film plane.
    Output:
        - rms_sq: mi.ScalarFloat. The square of the RMS radius of the spot.
    '''
    scaled_image = image / dr.mean(dr.detach(image))
    i = dr.arange(mi.Float, image.shape[0])
    j = dr.arange(mi.Float, image.shape[1])
    ii, jj = dr.meshgrid(i, j, indexing='ij')
    I = (scaled_image[:,:,0] + scaled_image[:,:,1] + scaled_image[:,:,2]) / 3.0
    inv_I_sum = dr.rcp(dr.sum(I))
    ibar = dr.detach(dr.sum(ii * I) * inv_I_sum)
    jbar = dr.detach(dr.sum(jj * I) * inv_I_sum)
    rms_sq = dr.sum(I * (dr.sqr(ii - ibar) + dr.sqr(jj - jbar))) * inv_I_sum
    return rms_sq

def rms_loss_and_center(image: mi.Color3f) -> tuple[mi.Float, mi.Float, mi.Float]:
    scaled_image = image / dr.mean(dr.detach(image))
    i = dr.arange(mi.Float, image.shape[0])
    j = dr.arange(mi.Float, image.shape[1])
    ii, jj = dr.meshgrid(i, j, indexing='ij')
    I = (scaled_image[:,:,0] + scaled_image[:,:,1] + scaled_image[:,:,2]) / 3.0
    inv_I_sum = dr.rcp(dr.sum(I))
    ibar = dr.detach(dr.sum(ii * I) * inv_I_sum)
    jbar = dr.detach(dr.sum(jj * I) * inv_I_sum)
    rms_sq = dr.sum(I * (dr.sqr(ii - ibar) + dr.sqr(jj - jbar))) * inv_I_sum
    return rms_sq, ibar, jbar

# def color_loss(image: mi.Color3f):
#     '''
#     Compute a loss that penalizes color dispersion.
#     '''
#     normalization = dr.prod(image.shape[:2]) * dr.max(dr.detach(image))
#     R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
#     return 0.5 * dr.rcp(normalization) * dr.sum(dr.sqr(R - G) + dr.sqr(R - B) + dr.sqr(G - B))