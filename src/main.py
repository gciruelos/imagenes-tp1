import numpy as np
from PIL import Image
from sys import argv
import utils

im1 = np.asarray(Image.open(argv[1]).convert('RGB'))
im2 = utils.to_hsi(im1)
im4 = utils.to_rgb(im2)
#side_by_side.sbys_histogram([im1, im2, im3, im4], ['rgb', 'hsi', 'hsi', 'rgb'],
#                                argv=argv[2] if len(argv)>2 else None)
