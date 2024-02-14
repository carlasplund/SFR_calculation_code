import matplotlib.pyplot as plt
import numpy as np
import scipy
import copy
import os


class Raw:
    def __init__(self, height=2200, width=3200, shift=4, fmt='<u2'):
        self.height = height
        self.width = width
        self.shift = shift
        self.fmt = fmt

    def read(self, path):
        image = np.fromfile(path, dtype=np.dtype(self.fmt)) >> self.shift
        return np.reshape(image, (self.height, self.width)).astype(float)

    def write(self, im_data, path):
        output = (im_data.flatten().astype(self.fmt) << self.shift).tobytes()
        with open(path, 'wb') as f:
            f.write(output)


def airy_disk(lam_diffr, f_number, grid_spacing, psf_shape):
    y_max, x_max = psf_shape
    xx, yy = np.meshgrid(np.arange(y_max), np.arange(x_max))
    r = np.sqrt((xx - 0.5 * x_max) ** 2 + (yy - 0.5 * y_max) ** 2) * grid_spacing
    r_prime = np.pi * r / (lam_diffr * f_number)
    r_prime = np.where(r_prime == 0, 1e-16, r_prime)
    psf_airy_disk = (2 * scipy.special.j1(r_prime) / r_prime) ** 2
    psf_airy_disk /= np.sum(psf_airy_disk)
    return psf_airy_disk


def abs_and_remove_padding(c, pad_width):
    return np.abs(c[pad_width:-pad_width, pad_width:-pad_width]) if pad_width > 0 else np.abs(c)


def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0], new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


def slanted_edge_blurred_with_diffraction_only(shape, pixel_pitch, f_number, wavelength,
                                               pad_width=10, oversampling=10):
    import slanted_edge_target

    roi_height, roi_width = (np.array(shape) + 2 * pad_width) * oversampling
    im_roi = slanted_edge_target.make_ideal_slanted_edge((roi_height, roi_width))

    grid_spacing = pixel_pitch / oversampling  # pixel_pitch is given in m

    def calc_psf_shape(lam_diffr, f_number, grid_spacing, n=3):
        # Calculate a psf_kernel size suitable for the psf blur size and pixel pitch
        sz = int(n * lam_diffr * f_number * 2.44 / grid_spacing)  # set as n x first minimum dia.
        return sz, sz

    psf_shape = calc_psf_shape(wavelength, f_number, grid_spacing)
    psf_diff = airy_disk(wavelength, f_number, grid_spacing, psf_shape)

    blurred_im_roi = scipy.signal.convolve2d(im_roi, psf_diff, mode='same')

    blurred_im_roi = rebin(blurred_im_roi, (np.array(blurred_im_roi.shape) / oversampling).astype(int))
    blurred_im_roi = abs_and_remove_padding(blurred_im_roi, pad_width)

    # try with Fourier transforms to speed up things
    ft_psf = np.fft.fftshift(np.fft.fft2(airy_disk(wavelength, f_number, grid_spacing, im_roi.shape)))
    ft_im_roi = np.fft.fftshift(np.fft.fft2(im_roi))
    blurred_im_roi_from_ft = np.abs(np.fft.fftshift(np.fft.ifft2(ft_psf * ft_im_roi)))
    blurred_im_roi_from_ft = rebin(blurred_im_roi_from_ft,
                                   (np.array(blurred_im_roi_from_ft.shape) / oversampling).astype(int))
    blurred_im_roi_from_ft = abs_and_remove_padding(blurred_im_roi_from_ft, pad_width)

    return blurred_im_roi, blurred_im_roi_from_ft


def extrap_mtf(input_lens_mtf, fit_begin=[60, 70], fit_end=[100, 90], mtf_fit_limit=[0.40],
               mtf_tail_lvl=0.05, extend_to_fit=True):
    """
    The system / sensor MTF curve is unreliable at high spatial frequency, where the sensor MTF is small.
    Therefore, we fit the lens MTF curve in a frequency interval and extrapolate down to zero MTF beyond that
    interval with a soft tail. If needed, more points added to fit the whole MTF curve down to ~zero MTF.

    :param input_lens_mtf: numpy array with spatial frequencies in cy/mm in the first row,
           and MTF (0.0-1.0) values in the second row, which are obtained by dividing the system MTF by
           the image sensor MTF (e.g. a sinc function)
    :param fit_begin: list of two alternative start points for the fit interval (cy/mm)
    :param fit_end: list of two alternative end points for the fit interval (cy/mm)
    :param mtf_fit_limit: list of one MTF level at which we switch from trying to use the first fit interval to the second
    :param mtf_tail_lvl: MTF beneath which the soft tail starts
    :param extend_to_fit: Add more points to fit the whole MTF curve (if necessary)
    :return:    2-d numpy array with spatial frequencies and lens MTF,
                [fit range start used, fit range end used]
    """

    f, mtf = input_lens_mtf[:, 0], input_lens_mtf[:, 1]
    mtf_ = scipy.interpolate.interp1d(f, mtf)

    k = 1 if mtf_(fit_end[0]) < mtf_fit_limit[0] else 0
    i = np.argwhere((fit_begin[k] <= f) & (f <= fit_end[k])).squeeze()
    slope, offset = np.polyfit(f[i].squeeze(), mtf[i].squeeze(), 1)

    x = copy.copy(f)
    run_look = True
    while run_look:
        y = offset + slope * x
        i0 = i[0]
        y[0:i0] = mtf[0:i0].squeeze()
        y[i0] = np.mean([mtf[i0], y[i0]])

        if any(y < mtf_tail_lvl):
            j0 = np.argwhere(y < mtf_tail_lvl)[0]
        else:
            j0 = len(y) - 1

        if extend_to_fit:
            if j0 < (len(y) - 10):
                run_look = False
            else:
                x_ext = x[1:11] - x[0] + x[-1]
                x = np.append(x, x_ext)
        else:
            run_look = False

    j = np.arange(j0, len(y))
    y[j] = y[j0] * (y[j0] / y[j0 - 1]) ** (j - j0)
    return np.column_stack([x, y]), [fit_begin[k], fit_end[k]]


def plot_image_and_crop_roi(fig, ax1, im, xc, yc, roi_height=80, roi_width=80):
    import selection_tools
    ax1.imshow(im, cmap='gray')
    r = selection_tools.RectXY(fig, ax1)
    fig.canvas.mpl_connect('motion_notify_event', r.move)
    fig.canvas.mpl_connect('button_press_event', r.fix_or_release)
    fig.canvas.mpl_connect('key_release_event', r.keypress)
    roi_data = [{'x_center': xc, 'y_center': yc, 'height': roi_height, 'width': roi_width}]
    r.populate_roi_data(roi_data)
    # (selection_tools supports using multiple ROIs on the same image, but here we only use one ROI per image)
    im_roi = selection_tools.crop(im, roi_data)[0]
    return im_roi


class MTFplotter:
    def __init__(self, pixel_pitch, f_number=2.8, lam_diffr=550e-9, x_lims=[0, 120],
                 fit_begin=[60, 70], fit_end=[100, 90], mtf_fit_limit=[0.40], mtf_tail_lvl=0.05):
        self.pixel_pitch = pixel_pitch
        self.f_number = f_number  # f-number for which we calculate diffraction
        self.lam_diffr = lam_diffr  # wavelength for which we calculate diffraction
        self.x_lims = x_lims  # plot limits for spatial frequency (cy/mm)
        self.fit_begin = fit_begin  # Two alternative start points for the fit interval
        self.fit_end = fit_end  # Two alternative end points for the fit interval
        self.mtf_fit_limit = mtf_fit_limit  # Where we switch from trying to use the first fit interval to the second
        self.mtf_tail_lvl = mtf_tail_lvl

    def calc_and_plot_mtf(self, ax, im_roi, x_lims=None):
        import utils
        import SFR
        # Calculate MTF from the ROI with the slanted edge
        sfr_lin = SFR.SFR(quadratic_fit=False)  # force fitting to a straight edge, as in the ISO 12233 standard
        mtf_lin, status = sfr_lin.calc_sfr(im_roi)
        sfr = SFR.SFR()  # allow fitting to a 2nd order polynomial edge shape
        mtf, status = sfr.calc_sfr(im_roi)

        f0 = 1000 / self.pixel_pitch
        mtf_lin[:, 0] *= f0
        mtf[:, 0] *= f0

        mtf_system = mtf
        # ideal sensor MTF: 100% fill factor, no crosstalk
        mtf_sinc = np.column_stack(
            [mtf_system[:, 0], np.abs(np.sinc(mtf_system[:, 0] / f0))])  # TODO: add support for fill factor
        mtf_lens_raw = np.column_stack([mtf_system[:, 0], mtf_system[:, 1] / (mtf_sinc[:, 1] + 1e-8)])

        # The system / sensor MTF curve is unreliable at high spatial frequency, where the sensor MTF is small.
        # Therefore, we fit the lens MTF curve in a frequency interval and extrapolate down to zero MTF beyond that
        # interval with a soft tail. If needed, more points added to fit the whole MTF curve down to ~zero MTF.

        mtf_lens, fit_range_used = \
            utils.extrap_mtf(mtf_lens_raw, fit_begin=self.fit_begin, fit_end=self.fit_end,
                             mtf_fit_limit=self.mtf_fit_limit, mtf_tail_lvl=self.mtf_tail_lvl, extend_to_fit=True)

        # Diffraction limit MTF for reference (valid at wavelength == lam_diffr)
        mtf_diff = np.column_stack([mtf_lens[:, 0],
                                    utils.mtf_diffraction_limit(self.f_number, self.lam_diffr, mtf_lens[:, 0])])

        ax.plot(mtf_lin[:, 0], mtf_lin[:, 1], '.:', color='C0', label="system MTF (lin. edge fit)")
        ax.plot(mtf_system[:, 0], mtf_system[:, 1], '.-', color='C1', label="system MTF")
        ax.plot(mtf_sinc[:, 0], mtf_sinc[:, 1], 'k-', label="ideal sensor MTF")
        ax.plot(mtf_lens_raw[:, 0], mtf_lens_raw[:, 1], '--', color='C2', label="system MTF / sensor MTF")
        ax.plot(mtf_lens[:, 0], mtf_lens[:, 1], '-.', color='C3',
                label=f"lens MTF, fitted btw {fit_range_used[0]:.0f} and {fit_range_used[1]:.0f} cy/mm")
        ax.plot(mtf_diff[:, 0], mtf_diff[:, 1], ':k',
                label=f'diffraction limit for {self.lam_diffr / 1e-9:.0f} nm, f/{self.f_number:.1f}')

        ax.set_xlim(*(x_lims if x_lims else self.x_lims))
        ax.set_ylim(0, 1.2)
        ax.grid()
        ax.set_ylabel('MTF')
        ax.set_xlabel('Spatial frequency (cy/mm)')
        ax.legend(loc='best')
        return mtf_system, mtf_lens, status


def plot_contour(data_h, data_v, range_lo, range_hi, lim_min, lim_av, text,
                 save_folder, radius0=38.0, radius1=55.0, save_to_file=True):
    """
    Plot the MTF values at a specific spatial frequency vs. horizontal and vertical field angles as a colored
    contour plot.
    :param data_h: np.array of three columns, 1st hor. field angle, 2nd vert. field angle, 3rd MTF value in hor. dir.
    :param data_v: as data_h, but with 3rd column containing MTF value measured in vert. direction
    :param range_lo: lowest MTF value to be plotted as a contour
    :param range_hi: highest MTF value to be plotted as a contour
    :param lim_min: specification limit for the min(horizontal MTF, vertical MTF) plot
    :param lim_av: specification limit for the (horizontal MTF + vertical MTF) / 2 plot
    :param text: text to be displayed in the plot title as basis for the figure save filename
    :param save_folder: where to save the figure
    :param radius0: radius of inner field angle circle (in degrees)
    :param radius1: radius of outer field angle circle (in degrees)
    :param save_to_file: save plots to file if True
    :return:
    """
    data_av = np.column_stack([data_h[:, :2], 0.5 * (data_h[:, 2] + data_v[:, 2])])
    data_min = np.column_stack([data_h[:, :2], np.minimum(data_h[:, 2], data_v[:, 2])])
    for g, direction, contrast_limit in zip([data_h, data_v, data_av, data_min], ['hor.', 'vert.', 'av.', 'min.'],
                                            [lim_min, lim_min, lim_av, lim_min]):
        if direction in ['hor.', 'vert.']:
            continue
        points = g[:, :2]
        values = g[:, 2]
        xx, yy = np.meshgrid(np.arange(-radius1, radius1 + 1), np.arange(-radius0, radius0 + 1))
        grid_points = np.column_stack([xx.flatten(), yy.flatten()])
        zz = scipy.interpolate.griddata(points, values, grid_points, method='linear')
        zz = zz.reshape(xx.shape)
        plt.figure()
        cs = plt.contourf(xx, yy, zz, levels=np.arange(range_lo, range_hi + 0.01, 0.05))
        colors = ['r' if z <= contrast_limit else 'k' for z in cs.levels]

        linestyles = 'solid'
        linewidths = [1.5 if (z / 0.05).astype(int) % 2 else 1.0 for z in cs.levels]
        cs2 = plt.contour(cs, levels=cs.levels, linestyles=linestyles, linewidths=linewidths, colors=colors)
        cbar = plt.colorbar(cs)
        cbar.add_lines(cs2)
        a = np.linspace(0, 2 * np.pi, 101)
        for r in [radius0, radius1]:
            plt.plot(r * np.cos(a), r * np.sin(a), 'k--')
        plt.plot(points[:, 0], points[:, 1], '.k')
        plt.gca().axis('equal')
        plt.ylim([-radius0, radius0])
        plt.xlim([-radius1, radius1])
        plt.xlabel('Horizontal field angle (°)')
        plt.ylabel('Vertical field angle (°)')
        txt = f'{text}, {direction} MTF'
        plt.title(txt)
        if save_to_file:
            # remove/replace unsuitable characters from the title text for use as a filename
            filename = txt.replace('/', '_').replace(' ', '_').replace(',', '').replace('.', '')
            fpath = os.path.join(save_folder, filename + '.png')
            dpi = 200
            plt.savefig(fpath, dpi=dpi)
            plt.close()
    if not save_to_file:
        plt.show()


def read_8bit(path):
    """
    Reads .bmp, .png, .jpg, etc. files
    Uses plt.imread(), which in turn calls PIL.Image.open()
    Note that the .pgm implementation of PIL is badly broken (more specifically, PGM P2 and P5 16-bit is broken,
    however, PGM P5 8-bit works), so we have dedicated functions for loading and saving images in these formats.

    From the Matplotlib documentation:

    The returned array has shape
        (M, N) for grayscale images.
        (M, N, 3) for RGB images.
        (M, N, 4) for RGBA images.

    PNG images are returned as float arrays (0-1). All other formats are returned as int arrays,
    with a bit depth determined by the file's contents.
    """
    return plt.imread(path).astype(float)


def read_pgm(file_path):
    """
    Read .pgm image file in either
        P2 (ASCII text) format, or
        P5 (either 8-bit unsigned, or big endian 16-bit unsigned binary) format
    input: file path
    output: 2-d numpy array of float
    """

    # Read header as binary file in order to avoid "'charmap' codec can't decode byte" errors
    with open(file_path, 'rb') as f:
        lines = []
        while len(lines) < 3:
            new_line = f.readline().strip().decode("ascii")
            if new_line[0] != '#':
                lines.append(new_line)
        image_data_start = f.tell()

    magic_number = lines[0]
    cols, rows = [int(v) for v in lines[1].split()]
    max_val = int(lines[2])

    if magic_number in 'P2':  # convert ASCII format (P2) data into a list of integers
        with open(file_path, 'r') as f:
            f.seek(image_data_start)  # read file again, but this time as a text file; skip the metadata lines
            lines = f.readlines()
        image_data = []
        for line in lines:  # skip the metadata lines
            image_data.extend([int(c) for c in line.split()])

    elif magic_number in 'P5':
        # Read and convert the binary format (P5) data into an array of integers
        fmt = 'u1' if max_val < 256 else '>u2'  # either 8-bit unsigned, or big endian 16-bit unsigned
        image_data = np.fromfile(file_path, offset=image_data_start, dtype=np.dtype(fmt))

    return np.reshape(np.array(image_data), (rows, cols)).astype(float)


def write_pgm(im_data, file_path, magic_number='P5', comment=''):
    """
    Write .pgm image file in either
        P2 (ASCII text) format, or
        P5 (either 8-bit unsigned, or big endian 16-bit unsigned binary) format, depending on max value in im_data
    im_data: image data as 2-d numpy array of float or int
    input: file_path
    magic_number: either 'P5' (binary data) or 'P2' (ASCII data), default is 'P5'
    comment: comment line to be added in the metadata section, default is None
    """

    im_data = im_data.astype(int)

    comment_line = '# ' + comment
    rows, cols = im_data.shape
    size = str(cols) + ' ' + str(rows)
    max_val = str(255) if np.max(im_data) < 256 else str(65535)

    meta_data_lines = [magic_number, comment_line, size, max_val]

    with open(file_path, 'w', newline='\n') as f:
        for line in meta_data_lines:
            f.write(line + '\n')

    def limit_line_length(im_data_string, char_limit=70):
        # Divide the string of image values (im_data_string) into a list of lines (output_lines). In order to comply
        # with the .pgm P2 format, the lines must not exceed 70 characters (char_limit) in length.
        i, j, output_lines = 0, char_limit, []
        len_im_data = len(im_data_string)
        while i < len_im_data:
            while (i + j + 1) > len_im_data or im_data_string[i + j] != ' ':
                j -= 1
                if (i + j) == len_im_data:
                    break
            output_lines.append(im_data_string[i:i + j] + '\n')
            i += j
            j = char_limit
        return output_lines

    if magic_number in 'P2':
        # append data in ASCII format
        image_data_string = ' '.join([str(d) for d in im_data.flatten()])  # string with values separated by blanks
        lines = limit_line_length(image_data_string)  # divide string into lines with a max length of 70 characters
        with open(file_path, 'a', newline='\n') as f:
            f.writelines(lines)

    elif magic_number in 'P5':
        # append either 8-bit unsigned, or big endian 16-bit unsigned image data
        fmt = 'u1' if int(max_val) < 256 else '>u2'
        output = im_data.astype(fmt).tobytes()
        with open(file_path, 'ab') as f:
            f.write(output)


def relative_luminance(rgb_image, rgb_w=(0.2126, 0.7152, 0.0722)):
    # Return relative luminance of image, based on sRGB MxNx3 (or MxNx4) input
    # Default weights rgb_w are the ones for the sRGB colorspace
    if rgb_image.ndim == 2:
        return rgb_image  # do nothing, this is an MxN image without color data
    else:
        return rgb_w[0] * rgb_image[:, :, 0] + rgb_w[1] * rgb_image[:, :, 1] + rgb_w[2] * rgb_image[:, :, 2]


def rgb2gray(im_rgb, im_0, im_1):
    """Flatten a Bayer pattern image by using the mean color channel signals in two flat luminance regions (one darker,
    one lighter). In addition, return also the estimated pedestal and estimated relative color gains.
    The underlying assumption is that the signal can be described as pedestal + color_gain * luminance (+ noise)

    Input:
        im_rgb: 2-d numpy array of float
            color image with a 2x2 color filter array (CFA) pattern, such as RGGB, RCCG, RYYB, RGBC, etc.
        im_0: 2-d numpy array of float
            a part of im_rgb with constant luminance
        im_1: 2-d numpy array of float
            another part im_rgb with a constant luminance which is different from that in im_0
    Output:
        a 2-d numpy array of float
            this is the flattened image, normalized as if all pixels had the same color, more specifically the color
            with the strongest color_gain of the four colors in the CFA
        pedestal: float
            the estimated pedestal from the solution to the overdetermined equation system
        rev_gain: 2x2 numpy array of float
            the estimated reverse gains of the four color filters in the CFA, normalized to the strongest of the
            color gains
    """

    c_0 = [np.mean(im_0[i::2, j::2]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    c_1 = [np.mean(im_1[i::2, j::2]) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]

    # Define and solve the following overdetermined equation system:
    # c_1 - pedestal = lum_ratio * (c_0 - pedestal)
    # Solve Ax = b for x, where x = [pedestal * (1 - lum_ratio), lum_ratio]
    A = np.array([[1, c_0[i]] for i in range(4)])
    b = np.array([c_1[i] for i in range(4)])
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    lum_ratio = x[1]  # luminance ratio between light and dark sides of the edge
    pedestal = x[0] / (1 - lum_ratio)  # estimate the pedestal that is added to the RGB image (not affected by RGB gain)

    # Estimate dark side RGB signal without the pedestal
    c_0_p = [np.mean([c_0[i] - pedestal, (c_1[i] - pedestal) / lum_ratio]) for i in range(4)]

    # Define a reverse gain image that will flatten the RGB image, and normalize it to the
    # green channel (i.e. as if all pixels were green in an RGB image)
    c_p_max = np.max(c_0_p)
    rev_gain = np.array([c_p_max / c_0_p[k] for k in range(4)])
    rev_gain_image = np.zeros_like(im_rgb)
    for k, (i, j) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        rev_gain_image[i::2, j::2] = rev_gain[k]

    # return flattened image
    return (im_rgb - pedestal) * rev_gain_image + pedestal, pedestal, rev_gain


def mtf_diffraction_limit(f_num, lam, f):
    """ Optical transfer function (OTF) for a diffraction limited lens. The OTF is calculated as the autocorrelation
    of a circular aperture.
    Paramters:
        f_num: float
            f-number of the lens aperture
        lam: float
            wavelength in m
        f: numpy array of float
            spatial frequencies in cy/mm
    """
    v = lam / 1e-3 * f * f_num
    v = v.clip(0.0, 1.0) if isinstance(v, np.ndarray) else np.min((v, 1.0))
    return 2 / np.pi * (np.arccos(v) - v * np.sqrt(1 - v ** 2))


def test():
    import matplotlib.pyplot as plt

    # Test calculation of diffraction limited MTF
    f_max = 600
    f = np.linspace(0, f_max, f_max + 1)  # spatial frequency in cy/mm
    f_num = 4.0  # f-number
    lam = 500e-9  # wavelength in m
    mtf = mtf_diffraction_limit(f_num, lam, f)
    plt.figure()
    plt.plot(f, mtf, '.-')
    plt.grid('both', 'both')
    plt.title(f'Diffraction limited MTF for {lam / 1e-9:.0f} nm wavelength and  f/{f_num}')
    plt.show()

    # Test reading image file in .pgm P2 format (ASCII string)
    plt.figure()
    file_path = "test_pgm_P2.pgm"
    im = read_pgm(file_path)
    plt.imshow(im, cmap='gray')
    plt.title(f'PGM P2 (ASCII) file: {file_path}')

    # Test writing / reading raw binary file
    Raw().write(im, "test_raw.raw")
    im2 = Raw(*im.shape).read("test_raw.raw")
    total_diff = np.sum(np.abs(im2 - im))
    print(f'Total difference between original and written/read raw file: {total_diff}')

    # Test writing/reading different .pgm formats
    write_pgm(im, "test_ascii.pgm", magic_number='P2', comment='P2 ASCII')
    write_pgm(im, "test_binary_uint8.pgm", magic_number='P5', comment='P5 binary uint8')
    write_pgm(im + 300, "test_binary_uint16.pgm", magic_number='P5', comment='P5 binary uint16')

    for file_path in ["test_ascii.pgm", "test_binary_uint8.pgm", "test_binary_uint16.pgm"]:
        plt.figure()
        im = read_pgm(file_path)
        plt.title(f'PGM file: {file_path}')
        plt.imshow(im, cmap='gray')
    plt.show()


if __name__ == '__main__':
    test()
