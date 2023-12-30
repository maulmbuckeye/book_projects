"""
ascii.py

A python program that convert images to ASCII art.

Author: Mahesh Venkitachalam
"""

import argparse

import numpy as np
from PIL import Image

# gray scale level values from:
# http://paulbourke.net/dataformats/asciiart/


def get_average_luminosity(image):
    """
    Given PIL Image, return average value of grayscale value
    """
    # get image as numpy array
    im = np.array(image)
    # get shape
    w, h = im.shape
    # get average
    return np.average(im.reshape(w * h))


class EncoderToAscii:
    # 70 levels of gray
    gscale1 = r"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    # 10 levels of gray
    gscale2 = '@%#*+=-:. '

    def __init__(self, more_levels):
        self.more_levels = more_levels

    def encode(self, value):
        if self.more_levels:
            gsval = self.gscale1[int((value * 69) / 255)]
        else:
            gsval = self.gscale2[int((value * 9) / 255)]
        return gsval


class Converter:

    file_name = ""

    def __init__(self, file_name, more_levels):
        self.file_name = file_name
        self.raw_h = 0
        self.raw_w = 0
        self.image = None
        self.height_of_cell = 0
        self.width_of_cell = 0
        self.encoder = EncoderToAscii(more_levels)

    def convert_image_to_ascii(self, cols, scale):
        """
        Given Image and dims (rows, cols) returns an m*n list of Images
        """

        # convert to grayscale
        self.image = Image.open(self.file_name).convert('L')

        self.raw_w, self.raw_h = self.image.size[0], self.image.size[1]
        print("input image dims: {} x {}".format(self.raw_w, self.raw_h))
        # compute width of tile
        self.width_of_cell = self.raw_w / cols
        # compute tile height based on aspect ratio and scale
        self.height_of_cell = self.width_of_cell / scale
        # compute number of rows
        rows = int(self.raw_h / self.height_of_cell)

        print("cols: {}, rows: {}".format(cols, rows))
        print("tile dims: {} x {}".format(self.width_of_cell, self.height_of_cell))

        # check if image size is too small
        if cols > self.raw_w or rows > self.raw_h:
            print("Image too small for specified cols!")
            exit(0)

        text_image = []
        for j in range(rows):
            text_image.append("")

            for i in range(cols):
                title = self.crop_image_for_tile_ij(i, j)
                avg = int(get_average_luminosity(title))
                text_image[j] += self.encoder.encode(avg)

        return text_image

    def crop_image_for_tile_ij(self,  i, j):
        x1, x2 = self.get_range(i, self.width_of_cell, self.raw_w)
        y1, y2 = self.get_range(j, self.height_of_cell, self.raw_h)

        img = self.image.crop((x1, y1, x2, y2))
        return img

    def get_range(self, index, length, max_length):
        r1 = int(index * length)
        r2 = int((index + 1) * length)
        # correct last tile
        r2 = min(r2, max_length)

        return r1, r2


# main() function
def main():
    # create parser
    desc_str = "This program converts an image into ASCII art."

    parser = argparse.ArgumentParser(description=desc_str)
    # add expected arguments
    parser.add_argument('--file', dest='img_file', required=True)
    parser.add_argument('--scale', dest='scale', required=False)
    parser.add_argument('--out', dest='out_file', required=False)
    parser.add_argument('--cols', dest='cols', required=False)
    parser.add_argument('--morelevels', dest='moreLevels', action='store_true')

    # parse args
    args = parser.parse_args()

    # set output file
    out_file = 'out.txt'
    if args.out_file:
        out_file = args.out_file
    # set scale default as 0.43 which suits a Courier font
    scale = 0.43
    if args.scale:
        scale = float(args.scale)
    # set cols
    cols = 80
    if args.cols:
        cols = int(args.cols)

    print('generating ASCII art...')
    # convert image to ascii txt
    c = Converter(args.img_file, args.moreLevels)
    aimg = c.convert_image_to_ascii(cols, scale)

    # open file
    f = open(out_file, 'w')
    # write to file
    for row in aimg:
        f.write(row + '\n')
    # cleanup
    f.close()
    print("ASCII art written to {}.".format(out_file))


# call main
if __name__ == '__main__':
    main()
