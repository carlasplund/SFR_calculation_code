import matplotlib.pyplot as plt
import numpy as np


def read_raw(path, height=2200, width=3200, rshift=4, fmt='<u2'):
    image = np.fromfile(path, dtype=np.dtype(fmt)) >> rshift
    return np.reshape(image, (height, width)).astype(float)


def write_raw(im_data, path, lshift=4, fmt='<u2'):
    output = (im_data.flatten().astype(fmt) << lshift).tobytes()
    with open(path, 'wb') as f:
        f.write(output)


def read_8bit(path):
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

    if magic_number in 'P2':  # Convert ASCII format (P2) data into a list of integers
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


def test():
    import matplotlib.pyplot as plt

    # Test reading .pgm P2 file:
    plt.figure()
    file_path = "test_pgm_P2.pgm"
    im = read_pgm(file_path)  # .pgm P2 format (ASCII string)
    plt.imshow(im, cmap='gray')
    plt.title(f'PGM P2 (ASCII) file: {file_path}')

    # Test writing / reading raw binary file
    im_shape = im.shape
    write_raw(im, "test_raw.raw")
    im2 = read_raw("test_raw.raw", *im_shape)
    total_diff = np.sum(np.abs(im2 - im))
    print(f'Total difference between original and written/read raw file: {total_diff}')

    # Test writing/reading different .pgm formats
    write_pgm(im, "test_ascii.pgm", magic_number='P2', comment='P2 ASCII!!')
    write_pgm(im, "test_binary_uint8.pgm", magic_number='P5', comment='P5 binary uint8!!')
    write_pgm(im + 300, "test_binary_uint16.pgm", magic_number='P5', comment='P5 binary uint16!!')

    for file_path in ["test_ascii.pgm", "test_binary_uint8.pgm", "test_binary_uint16.pgm"]:
        plt.figure()
        im = read_pgm(file_path)
        plt.title(f'PGM file: {file_path}')
        plt.imshow(im, cmap='gray')
    plt.show()


if __name__ == '__main__':
    test()
