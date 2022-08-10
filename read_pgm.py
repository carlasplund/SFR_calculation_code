"""
read_pgm.m

Read .pgm image file in P2 (ascii) format
input: file path
output: Numpy array of int32
"""

import numpy as np


def read_pgm(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    # This ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)
    # make sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2' 
    # Convert data into a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])
    return np.reshape(np.array(data[3:]), (data[1],data[0]))


if __name__ == '__main__':
    import matplotlib.pyplot as plt    
    
    # Usage example:
    im = read_pgm("Olesia/20220119_MV1.1_MTF_collimators/VR_x1860_y1475.pgm")
    plt.imshow(im, cmap='gray') 
    plt.show()
    