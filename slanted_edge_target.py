"""
slanted_edge_target.m - Create synthetic slanted edges with varying levels of sharpness and imperfections
Written in 2022 by Carl Asplund carl.asplund@eclipseoptics.com

To the extent possible under law, the author(s) have dedicated all copyright 
and related and neighboring rights to this software to the public domain worldwide. 
This software is distributed without any warranty.
You should have received a copy of the CC0 Public Domain Dedication along with 
this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.


make_ideal_slanted_edge()
Make an ideal slanted edge passing through the center of the image, with 
maximum theoretical sharpness for 100% fill factor pixels.

make_slanted_curved_edge()
Make a slanted edge passing through the center of the image, with support for
edge curvature, uneven illumination, and arbitrary edge profiles.
"""

import numpy as np
import SFR


def make_ideal_slanted_edge(image_shape=(100, 100), angle=5.0, low_level=0.20, hi_level=0.80,
                            pixel_fill_factor=1.0):
    """
    Make an ideal slanted edge passing through the center of the image, with maximum theoretical sharpness
    for 100% fill factor pixels

    Parameters:
        image_shape: tuple of int
            (height, width) in units of pixels
        angle: float
            edge angle relative to the vertical axis (degrees), default value is 5°
        low_level: float
            gray level (arbitrary units) at the dark side of the edge
        hi_level: float
            gray level (arbitrary units) at the bright side of the edge
        pixel_fill_factor: float
            pixel fill factor, 1.0 gives ideal edge, <1.0 gives aliasing, >1.0 gives blurred edge
    Returns:
        slanted edge image as a numpy array of float
    """

    height, width = image_shape
    xx, yy = np.meshgrid(range(0, width), range(0, height))
    x_midpoint = width / 2.0 - 0.5
    y_midpoint = height / 2.0 - 0.5

    # Calculate distance to edge (<0 means pixel is to the left of the edge, 
    # >0 means pixel is to the right)
    dist_edge = np.cos(-angle * np.pi / 180) * (xx - x_midpoint) + -np.sin(-angle * np.pi / 180) * (yy - y_midpoint)

    dist_edge /= np.sqrt(pixel_fill_factor)

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
                             illum_gradient_angle=75.0,
                             illum_gradient_magnitude=+0.05, esf=InterpolateESF([-0.5, 0.5], [0.0, 1.0]).f):
    """
    Make a slanted edge passing through the center of the image, with support for
    edge curvature, uneven illumination, and custom edge spread functions.

    Parameters:
        image_shape: tuple of int
            (height, width) in units of pixels
        angle: float
            c.w. angle (degrees) of slanted edge relative to vertical axis, valid range is [-90.0, 90.0],
            default value is 5°
        curvature: float
            curvature k(y) = f''(y) / (1 + f'(y)^2)^(3/2), where x = f(y) is a 2nd order polynomial describing the edge
        low_level: float
            gray level (arbitrary units) at dark side of the edge
        hi_level: float
            gray level (arbitrary units) at bright side of the edge
        black_lvl: float
            gray level corresponding got zero illumination (a.k.a. pedestal)
        illum_gradient_angle: float
            illumination gradient direction (degrees), 0 is downward, 90 is left, 180 is up
        illum_gradient_magnitude: float
            illumination gradient magnitude, relative illumination change between center of image and edge, can be
            either negative of positive
        esf: function
            function that returns the edge spread profile as function of distance (in units of
            pixels), with 0.0 at minus infinity, and 1.0 at positive infinity
    Returns:
        im: numpy array of floats
            slanted edge image (2-d)
        dist_edge: numpy array of floats
            distance to the edge from each pixel (2-d)
    """

    angle = np.clip(-angle, a_min=-90.0, a_max=90.0)

    inv_c = 1.0
    step_fctr = 0.0
    angle_offset = 0.0

    # The algorithms for the edge distance are made with near-vertical edges 
    # in mind. Temporarily rotate the image 90° if the edge is more than 45° 
    # from the vertical axis.
    if np.abs(angle) > 45.0:
        angle_offset = -90.0
        image_shape = image_shape[::-1]  # width -> height, and height -> width
    if angle > 45.0:
        step_fctr = -1.0
        inv_c = -1.0

    def midpoint(image_shape):
        return image_shape[0] / 2.0 - 0.5, image_shape[1] / 2.0 - 0.5

    y_midpoint, x_midpoint = midpoint(image_shape)

    # Describe the curved edge shape as a 2nd order polynomial
    slope = SFR.slope_from_angle(angle + angle_offset)
    p = SFR.polynomial_from_midpoint_slope_and_curvature(y_midpoint, x_midpoint, slope, curvature * inv_c)

    # Calculate distance to edge (<0 means pixel is to the left of the edge, 
    # >0 means pixel is to the right)
    dist_edge = SFR.calc_distance(image_shape, p, quadratic_fit=True)

    # Reverse step direction if edge angle is in the lower two quadrants (between 
    # 90 and 270 degrees)
    step_dir = -1 if np.cos(np.deg2rad(angle + step_fctr * angle_offset)) < 0 else 1

    # Assign a gray value from the supplied ESF function to each pixel based 
    # on its distance from the edge 
    im = low_level + (hi_level - low_level) * esf(step_dir * dist_edge)

    # If previously rotated, reverse rotation of image back to the original orientation
    if np.abs(angle) > 45.0:
        im = im.T[:, ::-1]  # rotate 90° right by transposing and mirroring
        image_shape = image_shape[::-1]  # width -> height, and height -> width
        y_midpoint, x_midpoint = midpoint(image_shape)

    # Apply illumination gradient
    if illum_gradient_magnitude != 0.0:
        slope_gradient = SFR.slope_from_angle(illum_gradient_angle - 90.0)
        p = SFR.polynomial_from_midpoint_slope_and_curvature(y_midpoint, x_midpoint,
                                                             slope_gradient, 0.0)
        illum_gradient_dist = SFR.calc_distance(image_shape, p, quadratic_fit=False)
        illum_gradient = 1 + illum_gradient_dist / (image_shape[0] / 2) * illum_gradient_magnitude
        im = np.clip((im - black_lvl) * illum_gradient, a_min=0.0, a_max=None) + black_lvl

    return im, dist_edge


def calc_custom_esf(x_length=5.0, x_step=0.01, x_edge=0.0, pixel_fill_factor=1.00,
                    pixel_pitch=1.0, sigma=0.2, show_plots=0):
    # Create a custom edge spread function (ESF) by convolution of three functions:
    #  - an ideal edge, 
    #  - a line spread function (LSF) representing the optics transfer function,
    #  - a pixel aperture
    # This is intended as an example. In e.g. situations where the image 
    # sensor has noticeable pixel crosstalk, the LSF (or MTF) of the pixel 
    # itself should be used, instead of a simple aperture box.

    def gauss(x, h=0.0, a=1.0, x0=0.0, sigma=1.0):
        return h + a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    w = pixel_pitch / 2 * np.sqrt(pixel_fill_factor)  # aperture width

    # 1-d position vector
    x = np.arange(-x_length / 2, x_length / 2, x_step)

    # ideal edge (step function)
    edge = np.heaviside(x - x_edge, 0.5)

    # optics line spread function (LSF)
    lsf = gauss(x, x0=x_edge, sigma=sigma)

    # pixel aperture (box filter)
    pixel = np.heaviside(x - (x_edge - w), 0.5) * np.heaviside((x_edge + w) - x, 0.5)

    # Convolve edge with lsf and pixel
    edge_lsf = conv(edge, lsf)
    edge_lsf_pixel = conv(edge_lsf, pixel)

    if show_plots >= 5:
        import matplotlib.pyplot as plt
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
    return x, edge_lsf_pixel


def rgb2gray(im_rgb_crop_dark, im_rgb_crop_light, im_rgb):
    r0 = np.mean(im_rgb_crop_dark[0::2, 0::2])
    r1 = np.mean(im_rgb_crop_light[0::2, 0::2])
    g0 = np.mean(im_rgb_crop_dark[0::2, 1::2])
    g1 = np.mean(im_rgb_crop_light[0::2, 1::2])
    b0 = np.mean(im_rgb_crop_dark[1::2, 1::2])
    b1 = np.mean(im_rgb_crop_light[1::2, 1::2])

    k_gr = (g1 - g0) / (r1 - r0)
    k_gb = (g1 - g0) / (b1 - b0)
    k_rb = (r1 - r0) / (b1 - b0)
    pedestal = np.mean([(g0 - k_gr * r0) / (1 - k_gr),
                        (g1 - k_gr * r1) / (1 - k_gr),
                        (g0 - k_gb * b0) / (1 - k_gb),
                        (g1 - k_gb * b1) / (1 - k_gb),
                        (r0 - k_rb * b0) / (1 - k_rb),
                        (r1 - k_rb * b1) / (1 - k_rb)])

    x = np.mean([(r1 - pedestal) / (r0 - pedestal),  # x = light / dark luminance ratio
                 (g1 - pedestal) / (g0 - pedestal),
                 (b1 - pedestal) / (b0 - pedestal)])

    gain_r = np.mean([r0 - pedestal, (r1 - pedestal) / x])
    gain_g = np.mean([g0 - pedestal, (g1 - pedestal) / x])
    gain_b = np.mean([b0 - pedestal, (b1 - pedestal) / x])

    gain_image = np.zeros_like(im_rgb)
    gain_image[0::2, 0::2] = 1 / gain_r
    gain_image[0::2, 1::2] = 1 / gain_g
    gain_image[1::2, 0::2] = 1 / gain_g
    gain_image[1::2, 1::2] = 1 / gain_b

    return (im_rgb - pedestal) * gain_image


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt


    def gray2rgb(image, pedestal, gain_r, gain_g, gain_b, cfa_rgb=np.array([[0, 1], [1, 2]])):
        im_rgb = np.zeros_like(image).astype(float)
        im_rgb[0::2, 0::2] = image[0::2, 0::2] * gain_r + pedestal
        im_rgb[0::2, 1::2] = image[0::2, 1::2] * gain_g + pedestal
        im_rgb[1::2, 0::2] = image[1::2, 0::2] * gain_g + pedestal
        im_rgb[1::2, 1::2] = image[1::2, 1::2] * gain_b + pedestal
        return im_rgb


    for pixel_fill_factor in [4.0, 1.0, 0.04]:
        # Create an ideal slanted edge image with default settings
        image_float = make_ideal_slanted_edge(pixel_fill_factor=pixel_fill_factor)

        # Display the image in 8 bit grayscale
        nbits = 8
        image_int = np.round((2 ** nbits - 1) * image_float.clip(0.0, 1.0)).astype(np.uint8)

        # TODO: plt.imshow and plt.imsave() with cmap='gray' doesn't interpolate properly(!), leaving histogram gaps and neighboring peaks, so we make an explicitly grayscale MxNx3 RGB image instead
        image_int = np.stack([image_int for i in range(3)], axis=2)
        plt.figure()
        plt.imshow(image_int)
        plt.title(f'Ideal slanted edge, fill factor={pixel_fill_factor}')

        # Save as an image file in the current directory
        current_dir = os.path.abspath(os.path.dirname(__file__))
        save_path = os.path.join(current_dir, "ideal_slanted_edge_example.png")
        plt.imsave(save_path, image_int, vmin=0, vmax=255, cmap='gray')

        # Test RGB decoding / white balancing
        im_gray_input = image_int[:, :, 0].astype(float)
        im_gray_input = np.random.poisson(im_gray_input * 1e1) / 1e1  # add noise
        plt.figure()
        plt.title('im_gray_input')
        plt.imshow(im_gray_input, cmap='gray', vmin=0.0, vmax=1.2 * np.max(im_gray_input))

        pedestal = 19.0
        gain_r, gain_g, gain_b = 1.1, 1.8, 1.3
        im_rgb = gray2rgb(im_gray_input, pedestal, gain_r, gain_g, gain_b)  # simulate RGB image
        plt.figure()
        plt.title('im_rgb_input')
        plt.imshow(im_rgb, cmap='gray')

        im_rgb_crop_dark = im_rgb[0:20, 0:20]
        im_rgb_crop_light = im_rgb[0:20, -20:-1]
        im_gray = rgb2gray(im_rgb_crop_dark, im_rgb_crop_light, im_rgb)  # calc. pedestal and white balance image
        plt.figure()
        plt.title('im_gray')
        plt.imshow(im_gray, cmap='gray', vmin=0.0, vmax=5.0)

    # --------------------------------------------------------------------------------
    # Create a curved edge image with a custom esf
    esf = InterpolateESF([-0.5, 0.5], [0.0, 1.0]).f  # ideal edge esf for pixels with 100% fill factor

    # arrays of positions and corresponding esf values
    x, edge_lsf_pixel = calc_custom_esf(sigma=0.8, pixel_fill_factor=1.0, show_plots=5)

    esf = InterpolateESF(x, edge_lsf_pixel).f  # a more realistic (custom) esf

    if 0:
        for angle in range(-90, 90 + 1, 10):
            image_float, _ = make_slanted_curved_edge((80, 100), illum_gradient_angle=45.0,
                                                      illum_gradient_magnitude=4 * 0.15, curvature=-2 * 0.001,
                                                      low_level=0.25, hi_level=0.70, esf=esf, angle=angle)

            # Display the image in 8 bit grayscale
            nbits = 8
            image_int = np.round((2 ** nbits - 1) * image_float.clip(0.0, 1.0)).astype(np.uint8)
            image_int = np.stack([image_int for i in range(3)], axis=2)
            plt.figure()
            plt.title(f"angle: {angle:.1f}°")
            plt.imshow(image_int)

    # Save as an image file in the current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(current_dir, "slanted_edge_example.png")
    plt.imsave(save_path, image_int, vmin=0, vmax=255, cmap='gray')

    plt.show()
    print("Finished!")
