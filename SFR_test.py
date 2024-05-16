import matplotlib.pyplot as plt
import numpy as np
import os.path
import time

import SFR
import slanted_edge_target
import utils


def test():
    # A kind of "verbosity" for plots, higher value means more plots:
    show_plots = 8  # setting this to 0 will result in no plots and much faster execution
    print(f"show_plots: {show_plots}")

    n = 100  # sample ROI size is n x n pixels

    n_well_fs = 10000  # simulated no. of electrons at full scale for the noise calculation
    output_fs = 1.0  # image sensor output at full scale

    def add_noise(sample_edge):
        np.random.seed(0)  # make the noise deterministic in order to facilitate comparisons and debugging
        return np.random.poisson(sample_edge / output_fs * n_well_fs) / n_well_fs

    # --------------------------------------------------------------------
    # Create a curved edge image with a custom esf for testing
    esf = slanted_edge_target.InterpolateESF([-0.5, 0.5],
                                             [0.0, 1.0]).f  # ideal edge esf for pixels with 100% fill factor

    # arrays of positions and corresponding esf values
    x, edge_lsf_pixel = slanted_edge_target.calc_custom_esf(sigma=0.3, pixel_fill_factor=1.0, show_plots=show_plots)
    # a more realistic (custom) esf
    esf = slanted_edge_target.InterpolateESF(x, edge_lsf_pixel).f

    image_float, _ = slanted_edge_target.make_slanted_curved_edge((n, n), curvature=0.001,
                                                                  illum_gradient_angle=0.0,
                                                                  illum_gradient_magnitude=0.0,
                                                                  low_level=0.25, hi_level=0.70, esf=esf, angle=5.0)
    im = image_float

    # Display the image in 8 bit grayscale
    nbits = 8
    image_int = np.round((2 ** nbits - 1) * im.clip(0.0, 1.0)).astype(np.uint8)
    image_int = np.stack([image_int for i in range(3)], axis=2)

    # Save slanted edge ROI as an image file in the current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(current_dir, "slanted_edge_example.png")
    plt.imsave(save_path, image_int, vmin=0, vmax=255, cmap='gray')

    # --------------------------------------------------------------------
    # (This is where you would load your own ROI image from file. Remember to
    # also remove gamma and to apply white balance if raw images are used from
    # an image sensor with a Bayer (color filter) pattern.)

    # Load slanted edge ROI image from from file
    im = plt.imread("slanted_edge_example.png")

    sample_edge = utils.relative_luminance(im)
    for simulate_noise in [False]:  # [False, True]:
        sample = add_noise(sample_edge) if simulate_noise else sample_edge

        if show_plots >= 6:
            # display the image in 8 bit grayscale
            nbits = 8
            image_int = np.round((2 ** nbits - 1) * sample.clip(0.0, 1.0)).astype(np.uint8)
            # plt.ishow and plt.imsave with cmap='gray' doesn't interpolate properly(!), so we
            # make an explicit grayscale sRGB image instead
            image_int = np.stack([image_int for i in range(3)], axis=2)
            plt.figure()
            plt.imshow(image_int)
            # plt.imshow(image_int, cmap='gray', vmin=0, vmax=255)

        print(" ")
        sfr_linear = SFR.SFR(quadratic_fit=False, verbose=True, show_plots=show_plots)
        sfr = SFR.SFR(verbose=True, show_plots=show_plots)
        mtf_linear, status_linear = sfr_linear.calc_sfr(sample)
        mtf, status = sfr.calc_sfr(sample)
        print(f"\nNow do the exact same two function calls, but without diagnostic"
              f" plots, to get the true execution speed of the SFR calculation"
              f" from the {n:d} x {n:d} pixel ROI image:")
        # This is how you would call the function in an automated script.
        # Remember that you can comment out the "@execution_timer"
        # decorators in the SFR.py module and skip verbosity (default is False):
        sfr_linear = SFR.SFR(quadratic_fit=False)
        sfr = SFR.SFR()

        def meas_execution_time(func, roi_image, n_repeats=50):
            t0 = time.time()
            for i in range(n_repeats):
                func(roi_image)
            t1 = (time.time() - t0) / n_repeats
            return t1, t1 / roi_image.size

        t, t_per_pixel = meas_execution_time(sfr_linear.calc_sfr, sample, n_repeats=50)
        print(f"SFR.calc_sfr() with straight edge fitting took {t:.3f} s to execute, "
              f"or {t_per_pixel / 1e-6:.1f} us per pixel.")

        t, t_per_pixel = meas_execution_time(sfr.calc_sfr, sample, n_repeats=50)
        print(f"SFR.calc_sfr() with curved edge fitting took {t:.3f} s to execute, "
              f"or {t_per_pixel / 1e-6:.1f} us per pixel.")

        print(" ")

        if show_plots >= 1:
            plt.figure()
            plt.plot(mtf_linear[:, 0], mtf_linear[:, 1], '.-', label="linear fit to edge")
            plt.plot(mtf[:, 0], mtf[:, 1], '.-', label="quadratic fit to edge")
            f = np.arange(0.0, 2.0, 0.01)
            mtf_sinc = np.abs(np.sinc(f))
            plt.plot(f, mtf_sinc, 'k-', label="sinc")
            plt.xlim(0, 2.0)
            plt.ylim(0, 1.2)
            textstr = f"Edge angle: {status_linear['angle']:.1f}°"
            props = dict(facecolor='w', alpha=0.5)
            ax = plt.gca()
            plt.text(0.65, 0.60, textstr, transform=ax.transAxes,
                     verticalalignment='top', bbox=props)
            plt.grid()
            shape = f'{sample_edge.shape[1]:d}x{sample_edge.shape[0]:d} px'
            if simulate_noise and n_well_fs > 0:
                n_well_lo, n_well_hi = np.unique(sample_edge)[[0, -1]] * n_well_fs
                snr_lo, snr_hi = np.sqrt([n_well_lo, n_well_hi])
                noise = f' SNR={snr_lo:.0f} (dark) and SNR={snr_hi:.0f} (bright)'
            else:
                noise = 'out noise'
            angle = status_linear['angle']
            plt.title(f'SFR from {shape:s} curved slanted edge\nwith{noise:s}, edge angle={angle:.1f}°')
            plt.ylabel('MTF')
            plt.xlabel('Spatial frequency (cycles/pixel)')
            plt.legend(loc='best')

    # _______________________________________________________________________________
    # Test with ideal slanted edge, result should be very similar to a sinc function (fourier transform of
    # a square aperture representing the image sensor pixel) (black curve)
    if True:
        ideal_edge = slanted_edge_target.make_ideal_slanted_edge((n, n), angle=85.0)

        for simulate_noise in [False, True]:
            sample = add_noise(ideal_edge) if simulate_noise else ideal_edge

            if show_plots >= 6:
                # display the image in 8 bit grayscale
                nbits = 8
                image_int = np.round((2 ** nbits - 1) * sample.clip(0.0, 1.0)).astype(np.uint8)
                # plt.imshow and plt.imsave with cmap='gray' doesn't interpolate properly(!), so we
                # make an explicit grayscale sRGB image instead
                image_int = np.stack([image_int for i in range(3)], axis=2)
                plt.figure()
                plt.imshow(image_int)

            mtf_list = []
            oversampling_list = [4, 6, 8]
            for oversampling in oversampling_list:
                sfr.set_oversampling(oversampling)
                mtf, status = sfr.calc_sfr(sample)
                mtf_list.append(mtf)

            if show_plots:
                plt.figure()
                for j, oversampling in zip(range(len(mtf_list)), oversampling_list):
                    plt.plot(mtf_list[j][:, 0], mtf_list[j][:, 1], '.-', label=f"oversampling: {oversampling:2d}")
                f = np.arange(0.0, 2.0, 0.01)
                mtf_sinc = np.abs(np.sinc(f))
                plt.plot(f, mtf_sinc, 'k-', label="sinc")
                plt.xlim(0, 2.0)
                plt.ylim(0, 1.2)
                textstr = f"Edge angle: {status['angle']:.1f}°"
                props = dict(facecolor='w', alpha=0.5)
                ax = plt.gca()
                plt.text(0.65, 0.60, textstr, transform=ax.transAxes,
                         verticalalignment='top', bbox=props)
                plt.grid()

                plt.ylabel('MTF')
                plt.xlabel('Spatial frequency (cycles/pixel)')
                shape = f'{ideal_edge.shape[1]:d}x{ideal_edge.shape[0]:d} px'
                if simulate_noise and n_well_fs > 0:
                    n_well_lo, n_well_hi = np.unique(ideal_edge)[[0, -1]] * n_well_fs
                    snr_lo, snr_hi = np.sqrt([n_well_lo, n_well_hi])
                    noise = f' SNR(dark) = {snr_lo:.0f} and SNR(bright) = {snr_hi:.0f}'
                else:
                    noise = 'out noise'
                plt.title(f'SFR from {shape:s} ideal slanted edge\nwith{noise:s}')
                plt.legend(loc='best')


if __name__ == "__main__":
    test()
    plt.show()
