import matplotlib.pyplot as plt
import numpy as np
import scipy
import os


def get_parameters_from_file_name(im_filename):
    basename = os.path.basename(im_filename)
    name_parts = basename.split('_')

    x = int(name_parts[0].split('x')[1])
    y = int(name_parts[1].split('y')[1])

    def is_in(search_strings, name):
        return any(s for s in search_strings if s in name)  # return true if any of the strings occur in name

    # The f-number may not be written in a consistent way in the folder name, so we employ a brute force parsing
    # method that accounts for several versions in use. :) Add new variants to this list as necessary.
    key_list = [
        {'str_variants': ['f1_8'], 'f_num': 1.8},
        {'str_variants': ['f2_8', 'f_2_8', 'F2_8'], 'f_num': 2.8},
        {'str_variants': ['F2'], 'f_num': 2.0},
        {'str_variants': ['F4'], 'f_num': 4.0},
        {'str_variants': ['f5_6', 'f_5_6', 'f_56', 'F5_6'], 'f_num': 5.6},
        {'str_variants': ['F8'], 'f_num': 8.0},
        {'str_variants': ['f11', 'f_11', 'F11'], 'f_num': 11.0},
        {'str_variants': ['f16', 'f_16', 'F16'], 'f_num': 16.0},
    ]

    f_number = np.nan
    for key in key_list:
        if is_in(key['str_variants'], basename):
            f_number = key['f_num']
            break

    return x, y, f_number


def select_ROI_and_calc_MTF(folder, save_folder, im_filename, pixel_pitch, read_fcn, orig_ext,
                            specification_freqs, dpi=200, lam_diffr=550e-9):
    import utils

    full_im_path = os.path.join(folder, im_filename)
    folder_basename = os.path.basename(folder)
    print(f'Processing {os.path.join(folder_basename, im_filename)}')

    im_roi = read_fcn(full_im_path)  # read image data
    if len(im_roi.shape) > 2:
        im_roi = im_roi[:, :, 0]  # TODO find a more general way to handle RGB and RGBA images ************************
    roi_height, roi_width = im_roi.shape

    x, y, f_number = get_parameters_from_file_name(im_filename)

    # Prepare MTF curve plot object
    mtf_plotter = utils.MTFplotter(pixel_pitch, f_number, lam_diffr=lam_diffr, fit_begin=[0.54, 0.63],
                                   fit_end=[0.90, 0.81], mtf_fit_limit=[0.40], mtf_tail_lvl=0.05)

    for x_lims, range_str in zip([(0, 120), (0, 450)], ['', '_0to450cymm']):
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 4))

        ax1.imshow(im_roi, vmin=0.0, vmax=1.0, cmap='gray')

        # Plot MTF curves and return MTF data
        mtf_system, mtf_lens, status = mtf_plotter.calc_and_plot_mtf(ax2, im_roi, x_lims=x_lims)

        angle = status['angle']  # edge angle relative to vertical
        suptitle = f'(x, y) = ({x:d}, {y:d}), {roi_width:d}x{roi_height:d} px, f/{f_number:.1f}, edge: {angle:.1f}°'
        fig.suptitle(suptitle)

        file_basename = os.path.join(save_folder, im_filename).split(orig_ext)[0] + f'ROI_h{roi_height}_w{roi_width}'
        file_image_save_name = file_basename + f'_mtf{range_str}.png'
        plt.savefig(file_image_save_name, dpi=dpi)

    file_data_save_name = file_basename + f'_mtf_system.txt'
    np.savetxt(file_data_save_name, mtf_system, fmt='%.4e')

    file_data_save_name = file_basename + f'_mtf_lens.txt'
    np.savetxt(file_data_save_name, mtf_lens, fmt='%.4e')

    # Obtain lens MTF at the design specification frequencies
    mtf_lens_interp = scipy.interpolate.interp1d(mtf_lens[:, 0], mtf_lens[:, 1])
    for spatial_freq in specification_freqs:
        mtf_at_sf = mtf_lens_interp(spatial_freq)[()]
        print(f'MTF at {spatial_freq:.0f} cy/mm is {mtf_at_sf * 100:.0f}%')


def test():
    import utils

    folder = 'edge_samples'
    im_names = os.listdir(folder)
    im_names = [f for f in im_names if '.txt' not in f]
    if any('.bmp' in f for f in im_names):
        read_fcn, orig_ext = utils.read_8bit, '.bmp'
    if any('.png' in f for f in im_names):
        read_fcn, orig_ext = utils.read_8bit, '.png'
    if any('.data' in f for f in im_names):
        read_fcn, orig_ext = utils.Raw(height=2200, width=3200, shift=4).read, '.data'
    im_names = [f for f in im_names if orig_ext in f]

    for im_name in im_names:
        save_folder = folder + '_eval'
        os.makedirs(save_folder, exist_ok=True)
        pixel_pitch = 5.0  # pixel pitch (µm) of the image sensor
        lam_diffr = 550e-9  # wavelength (nm) for which the diffraction limit MTF is calculated
        specification_freqs = [60.0, 90.0]  # spatial frequencies (cy/mm) of special interest for the MTF
        select_ROI_and_calc_MTF(folder, save_folder, im_name, pixel_pitch, read_fcn, orig_ext,
                                specification_freqs, lam_diffr=lam_diffr)


if __name__ == '__main__':
    test()
