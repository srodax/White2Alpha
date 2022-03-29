#!/usr/bin/env python3
import numpy as np
import fileinput
import sys
import argparse

#***********************************************************************
# Parsing arguments
#***********************************************************************
parser = argparse.ArgumentParser(description='Convert white into alpha of a column of hex colors of format "0xRRGGBB". Outputs "0xAARRGGBB".')
parser.add_argument('files', type=str, nargs='*', help='Files with colors to be converted.')
args = parser.parse_args()
#***********************************************************************
# reading data as argument or std input
data = np.loadtxt(fileinput.input(args.files), dtype=float)
#***********************************************************************

# Taken from [https://stackoverflow.com/a/68809469], [https://colab.research.google.com/drive/1iJ-3ZdJ822JlZgJ3ZtGCyClzT5X7Wk0t#scrollTo=lc0hG1PKrbV3]
def rgb_white2alpha(rgb, ensure_increasing=False, ensure_linear=False, lsq_linear=False):
    """
    Convert a set of RGB colors to RGBA with maximum transparency.

    The transparency is maximised for each color individually, assuming
    that the background is white.

    Parameters
    ----------
    rgb : array_like shaped (N, 3)
        Original colors.
    ensure_increasing : bool, default=False
        Ensure that alpha values increase monotonically.
    ensure_linear : bool, default=False
        Ensure alpha values increase linear from initial to final value.
    lsq_linear : bool, default=False
        Use least-squares linear fit for alpha.

    Returns
    -------
    rgba : numpy.ndarray shaped (N, 4)
        Colors with maximum possible transparency, assuming a white
        background.
    """
    # The most transparent alpha we can use is given by the min of RGB
    # Convert it from saturation to opacity
    alpha = 255. - np.min(rgb, axis=1)
    if lsq_linear:
        # Make a least squares fit for alpha
        indices = np.arange(len(alpha))
        A = np.stack([indices, np.ones_like(indices)], axis=-1)
        m, c = np.linalg.lstsq(A, alpha, rcond=None)[0]
        # Use our least squares fit to generate a linear alpha
        alpha = c + m * indices
        alpha = np.clip(alpha, 0, 1)
    elif ensure_linear:
        # Use a linearly increasing/decreasing alpha from start to finish
        alpha = np.linspace(alpha[0], alpha[-1], rgb.shape[0])
    elif ensure_increasing:
        # Let's also ensure the alpha value is monotonically increasing
        a_max = alpha[0]
        for i, a in enumerate(alpha):
            alpha[i] = a_max = np.maximum(a, a_max)
    alpha = np.expand_dims(alpha, -1)
    # Rescale colors to discount the white that will show through from transparency
    rgb = (rgb + alpha - 255)
    rgb = np.divide(rgb, alpha, out=np.zeros_like(rgb), where=(alpha > 0))*255
    rgb = np.clip(rgb, 0, 255)
    rgb = rgb.astype(int)
    # Concatenate our alpha channel
    argb = np.concatenate((alpha, rgb), axis=1)
    return argb

# printing the result to std out
#np.savetxt(sys.stdout, np.stack(datanew,axis=1), newline='\n')

