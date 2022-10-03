# -*- coding: utf-8 -*-
"""
SFR.m

Created on Sat Jan  2 21:08:00 2021

@author: casp

SFR.calc_sfr()
Calculate spatial frequency response (SFR), a.k.a. MTF, for a slanted edge.
* input: image patch with slanted edge to be analyzed (2-d Numpy array of float)
* input (optional): angle (degrees) of the edge, if not specified a fitted value will be used 
* input (optional): offset (px) of the edge, if not specified a fitted value will be used 
* output: MTF organized as a 2-d array where first column is spatial frequency, and second column contains MTF values
* output: fitted edge angle and offset 

SFR.find_optimized_slope()
Find edge angle (in e.g. cases where edge is noisy) that maximizes the 
area under the MTF curve out to a given spatial frequency.
* input: image patch with slanted edge to be analyzed (2-d Numpy array of float)
* input (optional): f_check, the spatial frequency (in units of cy/px) out to which the MTF curve is integrated
* input (optional): start angle (degrees), if not specified a fitted value will be used 
* output: angle (degrees) and offset (px) for the optimal edge, to be used as input for SFR.calc_sfr()
"""

import slanted_edge_target
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from execution_timer import execution_timer


def angle_from_slope(slope):
    return np.rad2deg(np.arctan(slope))


def slope_from_angle(angle):
    return np.tan(np.deg2rad(angle))


def centroid(arr, conv_kernel=3, win_width=5):
    height, width = arr.shape

    win = np.zeros(arr.shape)
    for i in range(height):
        win_c = np.argmax(np.abs(np.convolve(arr[i, :], np.ones(conv_kernel), 'same')))
        win[i, win_c - win_width:win_c + win_width] = 1.0

        # x = np.ones((height, 1)) * np.arange(width)  # TODO shall we continue to use this?
    x, _ = np.meshgrid(np.arange(width), np.arange(height))  # TODO or shall we use this instead?
    sum_arr = np.sum(arr * win, axis=1)
    sum_arr_x = np.sum(arr * win * x, axis=1)

    # The following division will result in nan for any row that lack an
    # edge transition:
    with np.errstate(divide='ignore', invalid='ignore'):
        return sum_arr_x / sum_arr  # divide-by-zero warnings are suppressed


def differentiate(arr, kernel):
    if len(arr.shape) == 2:
        # Use 2-d convolution, but with a one-dimensional (row-oriented) kernel
        out = scipy.signal.convolve2d(arr, [kernel], 'same', 'symm')
    else:
        # Input is a one-dimensional array
        out = np.convolve(arr, kernel, 'same')
        # The first element is not valid since there is no 'symm' option, 
        # replace it with 0.0 (thereby maintaining the input array size)
        out[0] = 0.0
    return out


def find_edge(centr, patch_shape, angle=None, show_plots=False, verbose=False):
    # Find 2nd and 1st order polynomials that best approximate the
    # edge shape given by the vector of LSF centroids supplied  in "centr"
    #
    # input
    # centr: centroid location for each row
    # patch_shape: tuple with (height, width) info about the patch
    # angle: force this edge angle (degrees from vertical line) to be used in the linear fit
    # output
    # pcoefs: 2nd order polynomial coefs from the least squares fit to the edge
    # [slope, offset]: polynomial coefs from the liner fit to the edge

    # Weed out positions in the vector of centroid values that 
    # contain nan or inf. These positions represent rows that lack
    # an edge transition. Remove also the first and last values.
    idx = np.where(np.isfinite(centr))[0][1:-1]

    # Find the location and direction of the edge by fitting a line to the 
    # centroids on the form x = y*slope + offset
    if angle is None:
        slope, offset = np.polyfit(idx, centr[idx], 1)
    else:
        slope = slope_from_angle(angle)
        offset = np.polyfit(idx, centr[idx] - slope * idx, 0)

    # pcoefs contains quadratic polynomial coefficients for the x-coordinate
    # of the curved edge as a function of the y-coordinate: 
    # x = pcoefs[0] * y**2 + pcoefs[1] * y + pcoefs[2]
    pcoefs = np.polyfit(idx, centr[idx], 2)

    if show_plots:
        verbose and print("showing plots!")
        fig, ax = plt.subplots()
        ax.plot(centr[idx], idx, '.-', label="centroids")
        ax.plot(np.polyval([slope, offset], idx), idx, '-', label="linear fit")
        ax.plot(np.polyval(pcoefs, idx), idx, '--k', label="quadratic fit")
        ax.set_xlim([0, patch_shape[1]])
        ax.set_ylim([0, patch_shape[0]])
        ax.set_aspect('equal', 'box')
        ax.legend(loc='best')
        ax.invert_yaxis()
        plt.show()

    return pcoefs, slope, offset


def midpoint_slope_and_curvature_from_polynomial(a, b, c, x0, x1):
    # Describe input 2nd degree polynomial f(x) = a*x**2 + b*x + c in
    # terms of midpoint, slope (at midpoint), and curvature (at midpoint)
    x_mid = (x1 + x0) / 2
    y_mid = a * x_mid ** 2 + b * x_mid + c
    # Calculated slope as first derivative of f(x) at x = x_mid
    slope = 2 * a * x_mid + b
    # Calculate the curvature as k(x) = f''(x) / (1 + f'(x)^2)^(3/2)    
    curvature = 2 * a / (1 + slope ** 2) ** (3 / 2)
    return x_mid, y_mid, slope, curvature


def polynomial_from_midpoint_slope_and_curvature(x_mid, y_mid, slope, curvature):
    # Calculate a 2nd degree polynomial f(x) = a*x**2 + b*x + c that passes
    # through the midpoint (x_mid, y_mid) with the given slope and curvature 
    a = curvature * (1 + slope ** 2) ** (3 / 2) / 2
    b = slope - 2 * a * x_mid
    c = y_mid - a * x_mid ** 2 - b * x_mid
    return [a, b, c]


def cubic_solver(a, b, c, d):
    # Solve the equation a*x**3 + b*x**2 + c*x + d = 0 for a 
    # real-valued root x by Cardano's method
    # (https://en.wikipedia.org/wiki/Cubic_equation#Cardano's_formula)

    p = (3 * a * c - b ** 2) / (3 * a ** 2)
    q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)
    # A real root exists if 4 * p**3 + 27 * q**2 > 0
    sr = np.sqrt(q ** 2 / 4 + p ** 3 / 27)
    t = np.cbrt(-q / 2 + sr) + np.cbrt(-q / 2 - sr)
    x = t - b / (3 * a)
    return x


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


# @execution_timer
def calc_distance(data_shape, p, quadratic_fit=False, verbose=False):
    # Calculate the distance (with sign) from each point (x, y) in the
    # image patch "data" to the slanted edge described by the polynomial p.
    # It is assumed that the edge is approximately vertically orientated
    # (between -45° and 45° from the vertical direction).
    # Distances to points to the left of the edge are negative, and positive 
    # to points to the right of the edge.
    x, y = np.meshgrid(range(data_shape[1]), range(data_shape[0]))

    verbose and print(f'quadratic fit: {str(quadratic_fit):s}')

    if not quadratic_fit or p[0] == 0.0:
        slope, offset = p[1], p[2]  # use linear fit to edge
        a, b, c = 1, -slope, -offset
        a_b = np.sqrt(a ** 2 + b ** 2)
        # TODO: Investigate why the projection fails for 45° edge angles
        # |ax+by+c| / |a_b| is the distance from (x,y) to the slanted edge:
        dist = (a * x + b * y + c) / a_b
    else:
        # Define a cubic polynomial equation for the y-coordinate
        # y0 at the point (x0, y0) on the curved edge that is closest to (x, y)
        d = -y + p[1] * p[2] - x * p[1]
        c = 1 + p[1] ** 2 + 2 * p[2] * p[0] - 2 * x * p[0]
        b = 3 * p[1] * p[0]
        a = 2 * p[0] ** 2

        if p[0] == 0.0:
            y0 = -d / c  # solution if edge is straight (quadratic term is zero)
        else:
            y0 = cubic_solver(a, b, c, d)  # edge is curved

        x0 = p[0] * y0 ** 2 + p[1] * y0 + p[2]
        dxx_dyy = np.array(2 * p[0] * y0 + p[1])  # slope at (x0, y0)
        r2 = dot([1, -dxx_dyy], [1, -dxx_dyy])
        # distance between (x, y) and (x0, y0) along normal to curve at (x0, y0)
        dist = dot([x - x0, y - y0], [1, -dxx_dyy]) / np.sqrt(r2)
    return dist


# @execution_timer
def project_and_bin(data, dist, oversampling, verbose=True):
    # p contains quadratic polynomial coefficients for the x-coordinate
    # of the curved edge as a function of the y-coordinate: 
    # x = p[0]*y**2 + p[1]*y + p[2]

    # Create a matrix "bins" where each element represents the bin index of the 
    # corresponding image pixel in "data":
    bins = np.round(dist * oversampling).astype(int)
    bins = bins.flatten()
    bins -= np.min(bins)  # add an offset so that bins start at 0

    esf = np.zeros(np.max(bins) + 1)  # Edge spread function
    cnts = np.zeros(np.max(bins) + 1).astype(int)
    data_flat = data.flatten()
    for b_indx, b_sorted in zip(np.argsort(bins), np.sort(bins)):
        esf[b_sorted] += data_flat[b_indx]  # Collect pixel contributions in this bin
        cnts[b_sorted] += 1  # Keep a tab of how many contributions were made to this bin

    # Calculate mean by dividing by the number of contributing pixels. Avoid
    # division by zero, in case there are bins with no content.
    esf[cnts > 0] /= cnts[cnts > 0]
    if np.any(cnts == 0):
        if verbose:
            print("Warning: esf bins with zero pixel contributions were found. Results may be inaccurate.")
            print(f"Try reducing the oversampling factor, which currently is {oversampling:d}.")
        # Try to save the situation by patching in values in the empty bins if possible
        patch_cntr = 0
        for i in np.where(cnts == 0)[0]:  # loop through all empty bin locations
            j = [i - 1, i + 1]  # indices of nearest neighbors
            if j[0] < 0:  # Is left neighbor index outside esf array?
                j = j[1]
            elif j[1] == len(cnts):  # Is right neighbor index outside esf array?
                j = j[0]
            if np.all(cnts[j] > 0):  # Now, if neighbor bins are non-empty
                esf[i] = np.mean(esf[j])  # use the interpolated value
                patch_cntr += 1
        if patch_cntr > 0 and verbose:
            print(f"Values in {patch_cntr:d} empty ESF bins were patched by interpolation between their respective "
                  f"nearest neighbors.")
    return esf


def peak_width(y, rel_threshold):
    # Find width of peak in y that is above a certain fraction of the maximum value
    val = np.abs(y)
    val_threshold = rel_threshold * np.max(val)
    indices = np.where(val - val_threshold > 0.0)[0]
    return indices[-1] - indices[0]


def filter_window(lsf, oversampling, lsf_centering_kernel_sz=9,
                  win_width_factor=1.5, lsf_threshold=0.10):
    # Calculate MTF using the LSF as input and use a window-function as filter 
    # to remove high frequency noise originating in regions far from the edge

    nn0 = 20 * oversampling  # sample range to be used for the FFT, intial guess
    mid = len(lsf) // 2
    i1 = max(0, mid - nn0)
    i2 = min(2 * mid, mid + nn0)
    nn = (i2 - i1) // 2  # sample range to be used, final 

    # Filter LSF curve with a uniform kernel to better find center and 
    # determine an appropriate Hann window width for noise reduction
    lsf_conv = np.convolve(lsf[i1:i2], np.ones(lsf_centering_kernel_sz), 'same')

    # Base Hann window half width on the width of the filtered LSF curve
    hann_hw = max(np.round(win_width_factor * peak_width(lsf_conv, lsf_threshold)).astype(int), 5 * oversampling)

    bin_c = np.argmax(np.abs(lsf_conv))  # center bin, corresponding to LSF max

    # Construct Hann window centered over the LSF peak, crop if necessary to
    # the range [0, 2*nn]
    crop_l = max(hann_hw - bin_c, 0)
    crop_r = min(2 * nn - (hann_hw + bin_c), 0)
    hann_win = np.zeros(2 * nn)  # default value outside Hann function
    hann_win[bin_c - hann_hw + crop_l:bin_c + hann_hw + crop_r] = \
        np.hanning(2 * hann_hw)[crop_l:2 * hann_hw + crop_r]
    return hann_win, 2 * hann_hw, [i1, i2]


def calc_mtf(lsf, hann_win, idx, oversampling, diff_ft):
    # Calculate spatial frequency response (from the unfiltered LSF)
    i1, i2 = idx
    mtf = np.abs(np.fft.fft(lsf[i1:i2] * hann_win))
    nn = (i2 - i1) // 2
    mtf = mtf[:nn]
    mtf /= mtf[0]  # normalize to zero spatial frequency 
    f = np.arange(0, oversampling / 2, oversampling / nn / 2)  # spatial frequencies (cy/px)
    mtf *= (1 / np.sinc(f / diff_ft)).clip(0.0, 10.0)  # TODO: explain this correction with a suitable comment
    return np.column_stack((f, mtf))


@execution_timer
def calc_sfr(image, oversampling=4, show_plots=False, offset=None, angle=None,
             difference_scheme='backward', verbose=True, return_fig=False,
             quadratic_fit=False, try_to_remove_gradient=False):
    """"
    Calculate spectral response function (MTF)
    """
    # TODO: do some form of gradient compensation (from uneven illumination), calc 2-d gradient from median-filtered bright side and apply, how to do with black-level)?
    # TODO: apply Hann (or Hamming) window before calculating centroids, or do a second pass after find_edge with windowing, centroids, and find_edge
    if difference_scheme == 'backward':
        diff_kernel = np.array([1.0, -1.0])
        diff_offset = -0.5
        diff_ft = 4
    elif difference_scheme == 'central':
        diff_kernel = np.array([0.5, 0.0, -0.5])
        diff_offset = 0.0
        diff_ft = 2

    removed_gradient = False
    while True:
        # Calculate centroids for the edge transition of each row
        sample_diff = differentiate(image, diff_kernel)
        centr = centroid(sample_diff) + diff_offset
    
        # Calculate centroids also for the 90° right rotated image
        image_rot90 = image.T[:, ::-1]  # rotate by transposing and mirroring
        sample_diff = differentiate(image_rot90, diff_kernel)
        centr_rot = centroid(sample_diff) + diff_offset
    
        # Use rotated image if it results in fewer rows without edge transitions
        if np.sum(np.isnan(centr_rot)) < np.sum(np.isnan(centr)):
            verbose and print("Rotating image by 90°")
            image, centr = image_rot90, centr_rot
            rotated = True
        else:
            rotated = False  # TODO: return information that image was rotated to caller
    
        # Finds polynomials that describes the slanted edge by least squares 
        # regression to the centroids:
        #  - pcoefs are the 2nd order fit coefficients
        #  - [slope, offset] are the first order (linear) fit coefficients for the same edge
        pcoefs, slope, offset = find_edge(centr, image.shape, angle=angle,
                                          show_plots=show_plots, verbose=verbose)
    
        pcoefs = [0.0, slope, offset] if not quadratic_fit else pcoefs
    
        # Calculate distance (with sign) from each point (x, y) in the
        # image patch "data" to the slanted edge
        dist = calc_distance(image.shape, pcoefs, quadratic_fit=quadratic_fit, verbose=verbose)
    
        esf = project_and_bin(image, dist, oversampling, verbose=verbose)  # edge spread function
    
        lsf = differentiate(esf, diff_kernel)  # line spread function
    
        hann_win, hann_width, idx = filter_window(lsf, oversampling)  # define window to be applied on LSF
    
        def compensate_gradient(image, dist, edge_width, show_plots):
            print("-------------------------------------------------------------")
            # idx_edge = (-edge_width / 2 < dist) & (dist < edge_width / 2)
    
            idx_left = dist < -edge_width / 2
            idx_right = dist > edge_width / 2
            if np.mean(image[idx_left]) < np.mean(image[idx_right]):
                idx_low, idx_hi = idx_left, idx_right
            else:
                idx_low, idx_hi = idx_right, idx_left
    
            import remove_gradient
            image_s, removed_gradient, idx_edge, rel_noise, bl, isf = \
                remove_gradient.remove_gradient(image, idx_low, idx_hi, dist=dist, 
                verbose=True, show_plots=show_plots)
    
            return image_s, removed_gradient
    
        mtf = calc_mtf(lsf, hann_win, idx, oversampling, diff_ft)
    
        if show_plots or return_fig:
            i1, i2 = idx
            nn = (i2 - i1) // 2
            lsf_sign = np.sign(np.mean(lsf[i1:i2] * hann_win))
            fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
            ax.plot(esf[i1:i2], 'b.-', label=f"ESF, oversampling: {oversampling:2d}")
            ax.plot(lsf_sign * lsf[i1:i2], 'r.-', label=f"{'-' if lsf_sign < 0 else ''}LSF")
            # ax.plot(lsf_conv, 'k:', label="conv LSF")
            ax.plot(hann_win * ax.axes.get_ylim()[1] * 1.1, 'g.-', label=f"Hann window, width: {hann_width:d}")
            ax.set_xlim(0, 2 * nn)
            ax2 = ax.twinx()
            ax2.get_yaxis().set_visible(False)
            ax.grid()
            ax.legend(loc='upper left')
            # ax2.legend(loc='upper right')
            ax.set_xlabel('Bin no.')
            if show_plots:
                plt.show()
    
        if try_to_remove_gradient:
            image_s, removed_gradient = compensate_gradient(
                image, dist, 0.5 * hann_width / oversampling, show_plots)
            image = image_s
        
        if not removed_gradient:
            break        


    angle = angle_from_slope(slope)
    if not return_fig:
        return mtf, angle, offset
    else:
        return mtf, angle, offset, fig, ax


def find_optimized_slope(image, oversampling=4, show_plots=False,
                         f_check=0.50, label='Integrated MTF', coarse_steps=False,
                         fine_steps=False, start_angle=None, verbose=False):
    """
    f_check: frequency limit (cy/px) for calculating the area under the MTF curve
    """
    mtf_original, angle_orig, _ = calc_sfr(image, oversampling=oversampling,
                                           show_plots=False,
                                           offset=None, angle=start_angle,
                                           verbose=False, quadratic_fit=False)
    mtf_area_vec = []
    if coarse_steps:
        width = 8
        step = 0.25
    elif fine_steps:
        width = 1.5
        step = 0.08
    else:
        width = 3
        step = 0.1

    angle_vec = np.arange(angle_orig - width / 2, angle_orig + width / 2, step)
    # remove angles around 0°, which would give spurious peaks
    angle_vec = [a for a in angle_vec if a < -1.0 or 1.0 < a]
    for angle in angle_vec:
        mtf, _, _ = calc_sfr(image, oversampling=oversampling,
                             show_plots=False, angle=angle, verbose=verbose,
                             quadratic_fit=False)
        f = mtf[:, 0]
        # integrate area under the MTF curve up to spatial frequency f_check:
        mtf_area_vec.append(np.trapz(mtf[f <= f_check, 1], f[f <= f_check]))

    # fit to polynomial and select the angle which gives the maximum area
    p = np.polyfit(angle_vec, mtf_area_vec, 4)
    y_fit = np.polyval(p, angle_vec)
    angle_final = angle_vec[np.argmax(y_fit)]

    mtf_q, _, _ = calc_sfr(image, oversampling=oversampling,
                           show_plots=True, angle=None, verbose=verbose,
                           quadratic_fit=True)
    mtf_area_q = np.trapz(mtf_q[f <= f_check, 1], f[f <= f_check])

    if show_plots:
        plt.figure(figsize=(8, 5), dpi=200)
        plt.plot(angle_vec, mtf_area_vec, '.-b', label='MTF area')
        plt.plot(angle_vec, y_fit, '--r', label='fit')
        plt.plot(angle_orig * np.array([1, 1]), [0.0, np.max(y_fit)], '-.b',
                 label=f'original slope {angle_orig:.2f}°')
        plt.plot(angle_final * np.array([1, 1]), [0.0, np.max(y_fit)], '--k',
                 label=f'optimized slope {angle_final:.2f}°')
        plt.plot(angle_final, mtf_area_q, 's', label='quadratic fit')
        plt.xlabel('Edge slope angle (deg.)')
        plt.ylabel(f'Area under MTF curve up to {f_check:.2f} cy/px')
        plt.grid('both', 'both')
        plt.title(f'{label:s}, {oversampling:d}x oversampling', fontsize=10)
        plt.legend(loc='best')
        plt.show()
    return angle_final, angle_orig


def relative_luminance(rgb_image, rgb_w=(0.2126, 0.7152, 0.0722)):
    # Return relative luminance of image, based on sRGB MxNx3 (or MxNx4) input
    # Default weights rgb_w are the ones for the SRGB colorspace
    if rgb_image.ndim == 2:
        return rgb_image  # do nothing, this is an MxN image without color data
    else:
        return rgb_w[0] * rgb_image[:, :, 0] + rgb_w[1] * rgb_image[:, :, 1] + rgb_w[2] * rgb_image[:, :, 2]


def main():
    import os.path
    
    show_plots = True
    show_plots2 = False
    oversampling = 4
    N = 100
    np.random.seed(0)
    im = plt.imread("slanted_edge_example.png")
    
        
    # --------------------------------------------------------------------------------
    # Create a curved edge image with a custom esf
    esf = slanted_edge_target.InterpolateESF([-0.5, 0.5], [0.0, 1.0]).f  # ideal edge esf for pixels with 100% fill factor
    
    x, edge_lsf_pixel = slanted_edge_target.calc_custom_esf(sigma=0.3, show_plots=show_plots2)  # arrays of positions and corresponding esf values
    esf = slanted_edge_target.InterpolateESF(x, edge_lsf_pixel).f  # a more realistic (custom) esf
    
    image_float = slanted_edge_target.make_slanted_curved_edge((100, 100), curvature=1*0.001, 
                                                               illum_gradient_angle=45.0, 
                                                               illum_gradient_magnitude=-1*-0.30, 
                                           low_level=0.25, hi_level=0.70, esf=esf, angle=5.0)
    im = image_float
    
    # im = slanted_edge_target.make_slanted_curved_edge((N, N), angle=1 * 5.0,
    #                                                   curvature=-1 * 0.001 * 100 / N,
    #                                                   illum_gradient_magnitude=1 * +0.3,
    #                                                   black_lvl=0.05)
    # im = slanted_edge_target.make_slanted_curved_edge((N, N), angle=1 * 5.0,
    #                                                   curvature=-1 * 0.001 * 100 / N,
    #                                                   illum_gradient_magnitude=1 * +0.3,
    #                                                   black_lvl=0.05)
    im = im[:, ::-1]
    
    # Display the image in 8 bit grayscale
    nbits = 8
    image_int = np.round((2 ** nbits - 1) * im.clip(0.0, 1.0)).astype(np.uint8)
    image_int = np.stack([image_int for i in range(3)], axis=2)
    if True or show_plots2:
        plt.imshow(image_int)
        plt.show()
    
        # Save as an image file in the current directory
        current_dir = os.path.abspath(os.path.dirname(__file__))
        save_path = os.path.join(current_dir, "slanted_edge_example.png")
        plt.imsave(save_path, image_int, vmin=0, vmax=255, cmap='gray')
    
    im = plt.imread("slanted_edge_example.png")
    
    sample_edge = relative_luminance(im)
    for simulate_noise in [False]:  # [False, True]:
        # simulate photon noise
        if simulate_noise:
            n_well_FS = 10000  # simulated no. of electrons at full scale for the noise calculation
            output_FS = 1.0  # image sensor output at full scale
            sample = np.random.poisson(sample_edge / output_FS * n_well_FS) / n_well_FS
        else:
            sample = sample_edge

        if show_plots2:
            # display the image in 8 bit grayscale
            nbits = 8
            image_int = np.round((2 ** nbits - 1) * sample.clip(0.0, 1.0)).astype(np.uint8)
            # plt.ishow and plt.imsave with cmap='gray' doesn't interpolate properly(!), so we
            # make an explicit grayscale sRGB image instead
            image_int = np.stack([image_int for i in range(3)], axis=2)
            plt.imshow(image_int)
            # plt.imshow(image_int, cmap='gray', vmin=0, vmax=255)
            plt.show()

        # mtf, _, _ = calc_sfr(sample, oversampling=oversampling, show_plots=False)
        mtf, _, _ = calc_sfr(sample, oversampling=oversampling, show_plots=show_plots2)
        mtf_rem_gr, _, _ = calc_sfr(sample, oversampling=oversampling, show_plots=show_plots2, try_to_remove_gradient=True)

        mtf_quadr, _, _ = calc_sfr(sample, oversampling=oversampling, angle=None,
                                   show_plots=show_plots2, quadratic_fit=True)
        mtf_quadr_rem_gr, _, _ = calc_sfr(sample, oversampling=oversampling, angle=None,
                                   show_plots=show_plots2, quadratic_fit=True, try_to_remove_gradient=True)
        
        if show_plots:
            plt.figure()
            plt.plot(mtf[:, 0], mtf[:, 1], '.-', label="linear fit to edge")
            plt.plot(mtf_rem_gr[:, 0], mtf_rem_gr[:, 1], '-', label="linear fit, removed grad.")
            plt.plot(mtf_quadr[:, 0], mtf_quadr[:, 1], '.-', label="quadratic fit to edge")
            plt.plot(mtf_quadr_rem_gr[:, 0], mtf_quadr_rem_gr[:, 1], '--', label="quad. fit to edge, removed grad.")
            f = np.arange(0.0, 2.0, 0.01)
            mtf_sinc = np.abs(np.sinc(f))
            plt.plot(f, mtf_sinc, 'k-', label="sinc")
            plt.xlim(0, 2.0)
            plt.ylim(0, 1.2)
            plt.grid()
            shape = f'{sample_edge.shape[1]:d}x{sample_edge.shape[0]:d} px'
            if simulate_noise and n_well_FS > 0:
                n_well_lo, n_well_hi = np.unique(sample_edge)[[0, -1]] * n_well_FS
                snr_lo, snr_hi = np.sqrt([n_well_lo, n_well_hi])
                noise = f' SNR={snr_lo:.0f} (dark) and SNR={snr_hi:.0f} (bright)'
            else:
                noise = 'out noise'
            plt.title(f'Simulated {shape:s} curved slanted edge\nwith{noise:s}')
            plt.ylabel('MTF')
            plt.xlabel('Spatial frequency (cycles/pixel)')
            plt.legend(loc='best')
            plt.show()

    # _______________________________________________________________________________
    if False:
        ideal_edge = slanted_edge_target.make_ideal_slanted_edge((N, N), angle=5.0)

        for simulate_noise in [False, True]:
            # simulate photon noise
            if simulate_noise:
                n_well = 10000  # simulated no. of electrons at full scale for the noise calculation
                sample = np.random.poisson(ideal_edge * n_well)
                sample = sample / n_well
            else:
                sample = ideal_edge

            if show_plots:
                # display the image in 8 bit grayscale
                nbits = 8
                image_int = np.round((2 ** nbits - 1) * sample.clip(0.0, 1.0)).astype(np.uint8)
                # plt.ishow and plt.imsave with cmap='gray' doesn't interpolate properly(!), so we
                # make an explicit grayscale sRGB image instead
                image_int = np.stack([image_int for i in range(3)], axis=2)
                plt.imshow(image_int)
                # plt.imshow(image_int, cmap='gray', vmin=0, vmax=255)
                plt.show()

            mtf_list = []
            oversampling_list = [4, 6, 8]
            for oversampling in oversampling_list:
                mtf, _, _ = calc_sfr(sample, oversampling=oversampling, show_plots=show_plots)
                mtf_list.append(mtf)
            # win_divider_list = [1, 3, 4]
            # for win_divider in win_divider_list:
            #     mtf, _, _ = calc_sfr(sample, win_divider=win_divider, show_plots=show_plots)
            #     mtf_list.append(mtf)

            if show_plots:
                plt.figure()
                for j, oversampling in zip(range(len(mtf_list)), oversampling_list):
                    plt.plot(mtf_list[j][:, 0], mtf_list[j][:, 1], '.-', label=f"oversampling: {oversampling:2d}")
                # for j, win_divider in zip(range(len(mtf_list)), win_divider_list):
                #     plt.plot(mtf_list[j][:, 0], mtf_list[j][:, 1], '.-', label=f"win_divider: {win_divider:2d}")
                f = np.arange(0.0, 2.0, 0.01)
                mtf_sinc = np.abs(np.sinc(f))
                plt.plot(f, mtf_sinc, 'k-', label="sinc")
                plt.xlim(0, 2.0)
                plt.ylim(0, 1.2)
                plt.grid()
                # plt.title("oversampling: " + str(oversampling))
                plt.ylabel('MTF')
                plt.xlabel('Spatial frequency (cycles/pixel)')
                shape = f'{ideal_edge.shape[1]:d}x{ideal_edge.shape[0]:d} px'
                if simulate_noise and n_well > 0:
                    n_well_lo, n_well_hi = np.unique(ideal_edge)[[0, -1]] * n_well
                    snr_lo, snr_hi = np.sqrt([n_well_lo, n_well_hi])
                    noise = f' SNR={snr_lo:.0f} (dark) and SNR={snr_hi:.0f} (bright)'
                else:
                    noise = 'out noise'
                plt.title(f'Simulated {shape:s} ideal slanted edge\nwith{noise:s}')
                plt.legend(loc='best')
                plt.show()

    if False:
        # Test angle optimization algorithm
        optimized_angle, _ = find_optimized_slope(sample, show_plots=show_plots)
        print(f'optimized slope: {optimized_angle:+3.2f}°')


if __name__ == "__main__":
    main()
