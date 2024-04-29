import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/chuah/mitsuba3-camera/build/python')
import drjit as dr
import mitsuba as mi

def compute_z(x, y, c, K, z0):
    ''' 
    Compute the sag function (z-coord) of a conic surface at the radial 
    coordinate r ** 2 = x ** 2 + y ** 2. The resultant z-value is expressed in 
    camera coordinates (i.e. z = 0 is the film; +z faces towards object space).

    This version of the function assumes mi/dr types are used for x, y.
    '''
    r2 = dr.sqr(x) + dr.sqr(y)
    safe_sqr = dr.clamp(1 - (1 + K) * dr.sqr(c) * r2, 0.0, dr.inf)
    z = z0 - r2 * c * dr.rcp(1 + dr.sqrt(safe_sqr))
    return z

def compute_z_np(x_, y_, c, K, z0):
    ''' 
    Compute the sag function (z-coord) of a conic surface at the radial 
    coordinate r ** 2 = x ** 2 + y ** 2. The resultant z-value is expressed in 
    camera coordinates (i.e. z = 0 is the film; +z faces towards object space).

    This version of the function assumes that numpy types are used for x, y.
    '''
    x = mi.Float(x_)
    y = mi.Float(y_)
    return compute_z(x, y, c, K, z0).numpy()


def fma(a, b, c):
    return a * b + c

class Asphere:
    def __init__(self, c, K, a_vec, r_elem, z0 = 0.0, dtype=np.double):
        self.c = dtype(c)
        self.K = dtype(K)
        self.r_elem = dtype(r_elem)
        self.num_terms = len(a_vec)
        self.a_vec = np.array(a_vec, dtype=dtype)
        # self.na_vec = np.array(a_vec, dtype=dtype) * (2 * (np.arange(self.num_terms) + 2))
        self.dtype = dtype
        self.origin = np.array([0, 0, z0], dtype=dtype)

    def z_conic(self, r):
        r_ = r.astype(self.dtype)
        safe_sqr = np.maximum(1 - (1 + self.K) * (self.c * r_) ** 2, np.zeros_like(r_))
        out = self.c * r_ ** 2 / (1 + np.sqrt(safe_sqr))
        return out

    def z_full(self, r):
        out = np.zeros_like(r, dtype=self.dtype)
        r2 = r.astype(self.dtype) ** 2
        for a in self.a_vec[::-1]:
            out = fma(out, r2, a)
        safe_sqr = np.maximum(1 - (1 + self.K) * (self.c) ** 2 * r2, np.zeros_like(r2))
        out = fma(out, r2, self.c / (1 + np.sqrt(safe_sqr)))
        out = r2 * out
        return out

    def lies_on_surface(self, p):
        if np.any(np.isinf(p)):
            return False

        r = np.sqrt(p[0] ** 2 + p[1] ** 2)
        return np.allclose(self.z_horner(r), p[2])
    
    # def z_conic_grad(self, r):
    #     r_ = r.astype(self.dtype)
    #     safe_sqr = np.maximum(1 - (1 + self.K) * (self.c * r_) ** 2, np.zeros_like(r_))
    #     out = self.c * r_ / np.sqrt(safe_sqr)
    #     return out
    
    # def z_full_grad(self, r):
    #     r_ = r.astype(self.dtype)
    #     r2 = r_ ** 2
    #     out = np.zeros_like(r, dtype=self.dtype)
    #     for a in self.na_vec[::-1]:
    #         out = fma(out, r2, a)
    #     safe_sqr = np.maximum(1 - (1 + self.K) * (self.c) ** 2 * r2, np.zeros_like(r2))
    #     out = fma(out, r2, self.c / np.sqrt(safe_sqr))
    #     out = r_ * out
    #     return out
        
    # def test_grad(self, f, fdot):
    #     r0 = 1.0 / self.c
    #     r = np.linspace(-r0, r0, 10, dtype=np.double)[1:-1]
    #     if self.dtype == np.single:
    #         epss = np.logspace(-6,-2)
    #     else:
    #         epss = np.logspace(-12,-4)
    #     errs = []
    #     for eps in epss:
    #         grad_fd = (f(r + eps) - f(r)) / (eps)
    #         grad_an = fdot(r)
    #         errs.append(np.linalg.norm(grad_fd - grad_an) / grad_an.size)

    #     plt.figure()
    #     plt.loglog(epss, errs)
    #     plt.loglog(epss, epss, 'k--')