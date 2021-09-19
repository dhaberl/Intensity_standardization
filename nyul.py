#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MEDICAL IMAGE INTENSITY STANDARDIZATION
Ref [1] Nyúl, László G., and Jayaram K. Udupa. "On standardizing the MR image intensity scale"
Magn Reson Med. (1999) Dec;42(6):1072-81.
Ref [2] Shah, Mohak, et al. "Evaluating intensity normalization on MRIs of human brain with multiple sclerosis"
Medical image analysis 15.2 (2011): 267-282.
Python implementation based on: 
https://github.com/sergivalverde/MRI_intensity_normalization
https://github.com/jcreinhold/intensity-normalization
Date: 2021-09-01 
Author: DAVID HABERL
"""

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm


def get_landmarks(img, percs):
    """Get the landmarks for a given image

    Args:
        img (numpy array): Input image to process. Shape: (height, width)
        percs (numpy array): Percentiles where to calculate the landmarks 

    Returns:
        landmarks (numpy array): Intensity values corresponding to the percentiles
    """

    landmarks = np.percentile(img, percs)
    return landmarks


def learn_standard_scale(data, i_min=1, i_max=99, i_s_min=1, i_s_max=100, l_percentile=10, u_percentile=90, step=10):
    """Determine the standard scale for a given set of images. This step is often referred to as "training" or
    "learning".

    Args:
        data (list): Set of input images, i.e. [img-1, img-2, ..., img-N] with img-1 as numpy array (height, width).
        i_min (int, optional): Minimum percentile value of the overall intensity range that defines the lower boundary
        of the intensity of interest range (see Ref [2]). Defaults to 1.
        i_max (int, optional): Maximum percentile value of the overall intensity range that defines the upper boundary
        of the intensity of interest range (see Ref [2]). Defaults to 99.
        
        Note bene: If we consider i_min=1 and i_max=99, then we effectively prune the lower and upper 1 percentile of
        the intensity values (see Ref [2]).
        The values falling outside these bounds are discarded as outliers.
        
        i_s_min (int, optional): Minimum intensity value corresponding to i_min. Defaults to 1.
        i_s_max (int, optional): Maximum intensity value corresponding to i_max. Defaults to 100.
        
        Note bene: The i_s_min/i_s_max values will define the minimum and maximum value of the target intensity range
        as the values corresponding to i_min/i_max will be mapped to i_s_min/i_s_max.
        Hence, they define the minimum and maximum of the standard scale.
        
        l_percentile (int, optional): Middle percentile value for lower bound (e.g., for deciles 10). Defaults to 10.
        u_percentile (int, optional): Middle percentile value for upper bound (e.g., for deciles 90). Defaults to 90.
        step (int, optional): Step for middle percentiles (e.g., for deciles 10). Defaults to 10.
        
        Note bene: Example intensity-landmark configuration:
        (1) i_min=1, i_max=99, l_percentile=10, u_percentile=90, step=10:
            [i_min, l_percentile, ..., step, ..., u_percentile, i_max] = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
            => referred to as "decile formulation (see Ref [2]). Default config.
        (2) i_min=1, i_max=99, l_percentile=10, u_percentile=90, step=5:
            [i_min, l_percentile, ..., step, ..., u_percentile, i_max] =
            [1, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 99]

    Returns:
        standard_scale (numpy array): Standard scale landmarks, i.e. the average landmark intensities (averaged over
        N images <=> len(data)).
        percs (numpy array): Intensity-landmark configuration that was used.
    """

    # Initialize intensity-landmark configuration
    percs = np.concatenate(([i_min], np.arange(l_percentile, u_percentile+1, step), [i_max]))
    
    # Initialize standard scale
    standard_scale = np.zeros(len(percs))

    # Process each image in order to build the standard scale
    for img in tqdm(data):    
        # Use thresholding approach to separate the background from the foreground (see Ref [2])
        img_thresh = img > img.mean()  # The overall mean intensity of the image is used as threshold
        masked = img[img_thresh > 0]
        landmarks = get_landmarks(masked, percs)
        min_p = np.percentile(masked, i_min)
        max_p = np.percentile(masked, i_max)
        f = interp1d([min_p, max_p], [i_s_min, i_s_max])
        landmarks = np.array(f(landmarks))
        standard_scale += landmarks
    standard_scale = standard_scale / len(data)
    
    return standard_scale, percs


def apply_standard_scale(input_image, standard_scale, percs, interp_type='linear'):
    """Transformation of a given input image to the standard scale. The intensities of the input image are normalized to
    the standard scale.

    Args:
        input_image (numpy array): Input image to normalize. Shape: (height, width)
        standard_scale (numpy array): Standard scale landmarks.
        percs (numpy array): Intensity-landmark configuration that was used to obtain the standard_scale.
        interp_type (str, optional): Interpolation type. Defaults to 'linear'.

    Returns:
        normalized_image (numpy array): Normalized input image
    """
    
    img_thresh = input_image > input_image.mean()
    masked = input_image[img_thresh > 0]
    landmarks = get_landmarks(masked, percs)
    f = interp1d(landmarks, standard_scale, kind=interp_type, fill_value='extrapolate')
    # Apply transformation to input image
    normalized_image = f(input_image)

    return normalized_image
