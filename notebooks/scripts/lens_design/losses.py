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
    I = scaled_image[:,:,0]     # TODO: use grayscale version of the image
    ibar = dr.sum(ii * I) / dr.sum(I)
    jbar = dr.sum(jj * I) / dr.sum(I)
    rms_sq = dr.sum(I * (dr.sqr(ii - ibar) + dr.sqr(jj - jbar))) * dr.rcp(dr.sum(I))
    return rms_sq