"""
slanted_edge_target.m

make_ideal_slanted_edge()
Make an ideal slanted edge passing through the center of the image, with 
maximum theoretical sharpness for 100% fill factor pixels
input: image_shape, as a tuple of (cols, rows)
input: edge angle relative to the vertical axis (degrees), default value is 5°
input: gray level for the dark side of the edge
input: gray level for the bright side of the edge
output: slanted edge image as a numpy array of floats

make_slanted_curved_edge()
Make a slanted edge passing through the center of the image, with support for 
edge curvature, uneven illumination, and arbitrary edge profiles
input: image_shape, as a tuple of (cols, rows)
input: edge angle relative to the vertical axis (degrees) at midpoint, default value is 5°
input: curvature k(y) = f''(y) / (1 + f'(y)^2)^(3/2), where x = f(y) is a 2nd order polynomial describing the edge    
input: gray level for the dark side of the edge
input: gray level for the bright side of the edge
input: black level, i.e. gray level corresponding to zero illumination
input: illumination gradient angle (degrees), 0 is down, 90 is left, 180 is up
input: illumination gradient magnitude, relative change between center of image and edge, can be either negative of positive
input: function that returns the edge profile as function of distance, with 0.0 at minus infinity, and 1.0 at positive infinity
output: slanted edge image as a numpy array of floats
"""

import numpy as np
import SFR


def make_ideal_slanted_edge(image_shape=(100, 100), angle=5.0, low_level=0.20, hi_level=0.80):
    # Return slanted edge image as a 2-d Numpy array of float.
    # angle: angle (degrees) of slanted edge relative to vertical axis
    # low_level, hi_level: gray levels on either side of edge

    height, width = image_shape
    xx, yy = np.meshgrid(range(0, width), range(0, height))
    x_midpoint = width / 2.0 - 0.5
    y_midpoint = height / 2.0 - 0.5

    # Calculate distance to edge (<0 means pixel is to the left of the edge, 
    # >0 means pixel is to the right)
    dist_edge = np.cos(angle*np.pi/180) * (xx - x_midpoint) + \
                -np.sin(angle*np.pi/180) * (yy - y_midpoint)

    return low_level + (hi_level - low_level) * (0.5 + dist_edge.clip(-0.5, 0.5))


def conv(a, b):
    # Perform convolution of arrays a and b in a way that preserves the 
    # values at the boundaries.
    # The output has the same size as a, and it is assumed that len(a) >= len(b)
    pad_width = len(b)
    a_padded = np.pad(a, pad_width, mode='edge')
    return np.convolve(a_padded, b, mode='same')[pad_width:-pad_width] / np.sum(b)
    

class InterpolateESF:
    def __init__(self, xp, yp):
        self.xp = xp
        self.yp = yp

    def f(self, x):
        # linear interpolation
        return np.interp(x, self.xp, self.yp, left=0.0, right=1.0)


def make_slanted_curved_edge(image_shape=(100, 100), angle=5.0, curvature=0.001,
                             low_level=0.25, hi_level=0.85, black_lvl=0.05,
                             illum_gradient_angle=90.0,
                             illum_gradient_magnitude=+0.05, esf=InterpolateESF([-0.5, 0.5], [0.0, 1.0]).f):

    # Return a slanted edge image in floating point format, with support for 
    # edge curvature, illumination gradients, and custom edge spread functions.   
    # image_shape: (height, width) tuple, in units of pixels
    # angle: angle (degrees) of slanted edge relative to vertical axis
    # curvature: k(y) = f''(y) / (1 + f'(y)^2)^(3/2), where x = f(y) is the equation of the edge
    # low_level, hi_level: gray levels on either side of edge
    # black_lvl: gray level corresponding to zero illumination
    # illum_gradient_angle: illumination gradient direction (degrees), 0 is downward, 90 is to the right
    # illum_gradient_magnitude: relative illumination change between center of image and edge
    # esf: user supplied edge spread function (or edge profile) that takes position (in units pf pixels) as input and goes from 0.0 (left) to 1.0 (right)
    
    height, width = image_shape
    x_midpoint = width / 2.0 - 0.5
    y_midpoint = height / 2.0 - 0.5
    
    # Describe the curved edge shape as a 2nd order polynomial
    slope = SFR.slope_from_angle(angle)
    p = SFR.polynomial_from_midpoint_slope_and_curvature(x_midpoint, y_midpoint, 
                                                         slope, curvature)
    
    # Calculate distance to edge (<0 means pixel is to the left of the edge, 
    # >0 means pixel is to the right)
    dist_edge = SFR.calc_distance(image_shape, p, quadratic_fit=True)
    
    # Reverse step direction if edge angle is in the lower two quadrants (between 
    # 90 and 270 degrees)
    step_dir = 1 if np.cos(np.deg2rad(angle)) >= 0 else -1
    
    # Assign a gray value to each pixel based on its distance from the edge 
    im = low_level + (hi_level - low_level) * esf(step_dir * dist_edge)
    
    # Apply illumination gradient
    if illum_gradient_magnitude != 0.0:
        slope_gradient = SFR.slope_from_angle(illum_gradient_angle - 90.0)
        p = SFR.polynomial_from_midpoint_slope_and_curvature(x_midpoint, y_midpoint, 
                                                             slope_gradient, 0.0)
        illum_gradient_dist = SFR.calc_distance(image_shape, p, quadratic_fit=False)    
        illum_gradient = 1 + illum_gradient_dist / (height / 2) * illum_gradient_magnitude    
        im = np.clip((im - black_lvl) * illum_gradient, a_min=0.0, a_max=None) + black_lvl
    
    return im


def calc_custom_esf(x_length=5.0, x_step=0.01, x_edge=0.0, fill_factor=1.00, 
                    pixel_pitch=1.0, sigma=0.2, show_plots=True):
    # Create a custom edge spread function (ESF) by convolution of three functions:
    #  - an ideal edge, 
    #  - a line spread function (LSF) representing the optics transfer function,
    #  - a pixel aperture
    # This is intended as an example. In e.g. situations where the image 
    # sensor has noticeable pixel crosstalk, the LSF (or MTF) of the pixel 
    # itself should be used, instead of a simple aperture box.
    
    def gauss(x, h=0.0, a=1.0, x0=0.0, sigma=1.0):
        return h + a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
    w = pixel_pitch / 2 * np.sqrt(fill_factor)  # aperture width

    # 1-d position vector
    x = np.arange(-x_length/2, x_length/2, x_step)
    
    # ideal edge (step function)
    edge = np.heaviside(x - x_edge, 0.5)  
    
    # optics line spread function (LSF)
    lsf = gauss(x, x0=x_edge, sigma=sigma)  

    # pixel aperture (box filter)
    pixel = np.heaviside(x - (x_edge-w), 0.5) * np.heaviside((x_edge+w) - x, 0.5)
    
    # Convolve edge with lsf and pixel
    edge_lsf = conv(edge, lsf)
    edge_lsf_pixel = conv(edge_lsf, pixel)
    
    if show_plots:
        plt.figure()
        plt.plot(x, edge, label='edge')
        plt.plot(x, lsf, label='LSF from optics')
        plt.plot(x, edge_lsf, label='ESF from optics')
        plt.plot(x, pixel, label='pixel aperture')
        plt.plot(x, edge_lsf_pixel, '--', label='ESF sampled by pixels')
        plt.grid(linestyle='--')
        plt.xlabel('Position (pixel pitch)')
        plt.ylabel('Signal value')
        plt.legend()
        plt.show()
    return x, edge_lsf_pixel
    

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt

    # Create an ideal slanted edge image with default settings
    image_float = make_ideal_slanted_edge((100, 100), angle=5.0)

    # Display the image in 8 bit grayscale
    nbits = 8
    image_int = np.round((2 ** nbits - 1) * image_float.clip(0.0, 1.0)).astype(np.uint8)
    
    # TODO: plt.ishow and plt.imsave with cmap='gray' doesn't interpolate properly(!), leaving histogram gaps and neighboring peaks, so we
    # make an explicitly grayscale MxNx3 RGB image instead
    image_int = np.stack([image_int for i in range(3)], axis=2)
    plt.imshow(image_int)
    plt.show()
    
    # Save as an image file in the current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(current_dir, "ideal_slanted_edge_example.png")
    plt.imsave(save_path, image_int, vmin=0, vmax=255, cmap='gray')
    
    # --------------------------------------------------------------------------------
    # Create a curved edge image with a custom esf
    esf = InterpolateESF([-0.5, 0.5], [0.0, 1.0]).f  # ideal edge esf for pixels with 100% fill factor
    
    x, edge_lsf_pixel = calc_custom_esf(sigma=0.8)  # arrays of positions and corresponding esf values
    esf = InterpolateESF(x, edge_lsf_pixel).f  # a more realistic (custom) esf
    
    image_float = make_slanted_curved_edge((100, 100), illum_gradient_magnitude=0.1, 
                                           esf=esf, angle=5.0)

    # Display the image in 8 bit grayscale
    nbits = 8
    image_int = np.round((2 ** nbits - 1) * image_float.clip(0.0, 1.0)).astype(np.uint8)
    image_int = np.stack([image_int for i in range(3)], axis=2)
    plt.imshow(image_int)
    plt.show()
    
    # Save as an image file in the current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(current_dir, "slanted_edge_example.png")
    plt.imsave(save_path, image_int, vmin=0, vmax=255, cmap='gray')
    
    print("Finished!")


