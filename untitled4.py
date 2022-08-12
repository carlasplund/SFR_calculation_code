# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:51:59 2022

@author: casp
"""
import numpy as np
from execution_timer import execution_timer
import slanted_edge_target


@execution_timer
def fit_plane(z, pts):
    # Fit a plane to data supplied in 2-d array z. Use only array indices specified in pts.
    # TODO: Discard np.nan and other infinite values in z. 

    # pts = np.isfinite(z)  # these array positions will be discarded
    y_max, x_max = z.shape
    x, y = np.meshgrid(np.arange(0, x_max), np.arange(0, y_max))
    xx, yy = x[pts], y[pts]
    A = np.column_stack((xx, yy, np.ones(xx.shape)))

    # least square fit to plane
    coefs = np.linalg.lstsq(A, z[pts], rcond=None)[0]
    plane_fit = coefs[0] * x + coefs[1] * y + coefs[2]
    # residual divided by number of data points
    res = np.linalg.norm(z[pts] - plane_fit[pts]) / z.size
    return coefs, res


def remove_gradient(image, idx_lo, idx_hi, dist=None, allowed_gradient=1e-4,
                    step_factor_limit=1.1, verbose=False, show_plots=False):
    # Analyze the supplied slanted edge for signs of a illumination gradient. 
    # If present, remove it and return the flattened image.
    # image: numpy aray with slanted edge image
    # idx_lo: pixel indices for the dark side of the edge (excluding the edge transition region itself)
    # idx_hi: pixel indices for the bright side of the edge
    # allowed_gradient: magnitude of relative 2-d gradient that is accepted without correction
    # TODO is this input necessary? step_factor_limit:
    # TODO assumption: edge is near vertical (not horizontal)
    rows, cols = image.shape
    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
    idx_edge = ~(idx_lo | idx_hi)

    idx = {'lo': idx_lo,
           'hi': idx_hi}

    coefs, res, nonuniform_illum = {}, {}, {}
    for side in ['hi', 'lo']:
        coefs[side], res[side] = fit_plane(image, idx[side])

        # Check if gradients along x and/or y-directions is larger than the
        # allowed value
        # coefs consists of [x-coefficient, y-coefficient, constant]
        nonuniform_illum[side] = \
            np.linalg.norm(coefs[side][:2] / coefs[side][2]) > allowed_gradient

    verbose and print("coefs['hi']:", np.array_str(coefs['hi'], precision=2))
    verbose and print("coefs['lo']:", np.array_str(coefs['lo'], precision=2))

    # Estimate illumination step factor ('isf') by comparing the linear 
    # coefficients in x and y for the 'hi' and 'lo' parts
    isf = np.linalg.norm(coefs['hi'][:2]) / np.linalg.norm(coefs['lo'][:2])
    # total step factor, incl. black level
    tsf = np.linalg.norm(coefs['hi'][2]) / np.linalg.norm(coefs['lo'][2])
    verbose and print(f"isf: {isf:f}, tsf: {tsf:f}")

    if nonuniform_illum['lo'] and nonuniform_illum['hi'] and isf > step_factor_limit:
        # Estimate  camera black_level
        hi_lo = coefs['hi'][2] - coefs['lo'][2]
        bl = coefs['hi'][2] - isf / (isf - 1) * hi_lo  # estimated black level

        verbose and print(f"Estim. step factor: {isf:.3f}, estim. camera black level: {bl:.3f}")

        # # Compensated (gradient removed) image:
        # image_s = image * 1.0  # TODO: copy image in a nicer way
        # # image_s[idx_edge] = 0.0
        # for side in ['lo', 'hi']:
        #         image_s[idx[side]] = image_s[idx[side]] - \
        #                          coefs[side][0] * xx[idx[side]] - coefs[side][1] * yy[idx[side]]
                                 
        # Compensated (gradient removed) image:
        
        esf = slanted_edge_target.InterpolateESF([-0.5, 0.5], [0.0, 1.0]).f
        f = {'lo': 1.0 + (0.0 - 1.0) * esf(dist),
             'hi': 0.0 + (1.0 - 0.0) * esf(dist)}
        image_s = np.zeros(image.shape)
        for side in ['lo', 'hi']:
            image_s += f[side] * (image - coefs[side][0] * xx - coefs[side][1] * yy)
                                 
    else:
        image_s = image * 1.0  # TODO: copy image in a nicer way
        bl = None
        verbose and print(f"Estim. step factor: {isf:.3f}, camera black level could not be determined")

    rel_noise = np.std(image_s[idx['hi']]) / np.mean(image_s[idx['hi']])

    verbose and print(f"Noise: {rel_noise * 100:.2f}%")

    import matplotlib.pyplot as plt
    plt.figure()
    if show_plots:
        for ii in [0, rows - 1]:
            # plt.plot(xx[ii, :], zz_opt[ii, :], '-', label='zz_opt')
            plt.plot(xx[ii, :], image[ii, :], '-', label=f'image, row {ii:d}')
            # # plt.plot(xx[ii, :], zz_orig[ii, :], '.-', label='zz_orig')
            # plt.plot(xx[ii, :], zz_fit[ii, :], '-', label='zz_fit')
            if nonuniform_illum['hi']:
                # plt.plot(xx[ii, :], zz_corrected[ii, :], '-', label='zz_corrected')  
                plt.plot(xx[ii, :], image_s[ii, :], '-', linewidth=1.5, label=f'image_s, row {ii:d}')
                # plt.plot(xx[ii, :], zz_comp[ii, :], '-', label='zz_comp')  
                title = f"black_level: {bl:.2f}, illum step factor: {isf:.2f}, noise: {rel_noise * 100:.2f}%"
            else:
                title = f"No gradient correction was made, noise: {rel_noise * 100:.2f}%"
            plt.title(title)
            plt.grid('both')
            plt.legend()
        plt.show()

    return idx_edge, rel_noise, bl, isf


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    xx, yy = np.meshgrid(np.arange(100), np.arange(80))

    x_mid = 46
    edge_width = 10

    pts = {'lo': xx < x_mid - edge_width // 2,
           'hi': xx > x_mid + edge_width // 2}

    aa, bb, cc = 0.3, -0.5, 100.0
    aa *= 0.3
    bb *= 0.3
    light = aa * xx + bb * yy + cc

    edge = np.ones(light.shape)
    edge[xx < x_mid] = edge[xx < x_mid] / 4.0
    black_level = 20.0

    zz = light * edge + black_level

    n_sigma = 0.005
    np.random.seed(0)
    zz *= 1 + n_sigma * np.random.randn(*zz.shape)

    remove_gradient(zz, pts['lo'], pts['hi'], allowed_gradient=1e-4,
                    step_factor_limit=1.1, verbose=True, show_plots=True)
