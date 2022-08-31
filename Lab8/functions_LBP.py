#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:59:47 2018
Updated on Wed May 05 17:45:00 2021

@author: victor
"""

import numpy as np
import cv2


def get_limits(image, R):
    """
    This function defines the area in which the pixels from which the LBP codes are going to be extracted.
    Specifically, these pixels will be those in which the neighbourhood is located inside the image.

    Parameters
    ----------
    image : numpy array
        The image the LBP codes are going to be extracted.
    R : int
        Radius of the LBP

    Returns
    -------
    x_min, x_max, y_min, y_max : int
        Minimum and maximum column and minimum and maximum row, respectively, of the pixels the LPB codes can be
        extracted from.
    """


    # ====================== YOUR CODE HERE ======================

    # ============================================================

    return x_min, x_max, y_min, y_max

# ****************************************************************************


def vec_comps(image, i, j, P):
    """
    This function computes the binary code by comparing the central pixel with all the neighbourhood pixels. We are
     considering that the neighbour pixels are in the neighbourood of R=1 and P=8.

    Parameters
    ----------
    image : numpy array
        The image the LBP codes are going to be extracted.
    i, j : int
        Coordinates of the pixel the LBP is going to be extracted from.
    P : int
        Number of nighbourhood pixels.

    Returns
    -------
    vector01 : numpy boolean vector
        Vector which represents each point of the neighbourhood with 1 or 0 depending on the comparison of the grey
        level with the central pixel located at image(i, j).
    """

    vector01 = np.empty(P, dtype=bool)

    # ====================== YOUR CODE HERE ======================

    # ============================================================

    return vector01

# ****************************************************************************


def lbp_value(vector01):
    """
    This function computes the LBP value from a given LBP binary vector

    Parameters
    ----------
    vector01 : numpy boolean vector
        Vector which the LBP binary code (i.e., the comparison of the grey level with the central pixel).

    Returns
    -------
    val_lbp : int
        LBP value of the given binary LBP code
    """

    # ====================== YOUR CODE HERE ======================

    # ============================================================

    return val_lbp

# ****************************************************************************


def lbpri_value(vector01):
    """
    This function computes the LBP rotation invariant value from a given LBP binary vector. To make it rotation
    invariant, the comparisons of the values obtained with all possible bit shifts must be assessed, and the minimum is
    returned

    Parameters
    ----------
    vector01 : numpy boolean vector
        Vector which the LBP binary code (i.e., the comparison of the grey level with the central pixel).

    Returns
    -------
    val_lbp : int
        Rotation invariant LBP value of the given binary LBP code
    """

    # ====================== YOUR CODE HERE ======================

    # ============================================================

    return val_lbp

# ****************************************************************************


def lbpriu_value(vector01):
    """
    This function computes the uniform LBP value from a given LBP binary vector. It counts the number of bit
    transitions 0->1 or 1->0

    Parameters
    ----------
    vector01 : numpy boolean vector
        Vector which the LBP binary code (i.e., the comparison of the grey level with the central pixel).

    Returns
    -------
    val_lbp : int
        Uniform LBP value of the given binary LBP code
    """

    # ====================== YOUR CODE HERE ======================

    # ============================================================

    return val_lbp

# ****************************************************************************


def LBP_image(image):
    """
    This function computes an image which contains the LBP values of each pixel in which the LBP is calculated

    Parameters
    ----------
    image : numpy array
        Image from which the LBP values are going to be extracted

    Returns
    -------
    lbp_im_crop : numpy array
        Image with the LBP values of each pixel of the original image. It is cropped so that only the pixels from which
        the LBP has been calculated are shown.
    """

    R = 1
    P = 8

    # 1. If it is a colour image, convert it into grayscale
    # ====================== YOUR CODE HERE ======================

    # ============================================================

    # 2. Obtain the limits of the image taking into account the valid points
    x_min, x_max, y_min, y_max = get_limits(image, R)

    # 3. Obtain the LBP value for each valid pixel
    
    # Initialise matrix for the lbp values
    lbp_im = np.zeros(image.shape)

    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            vector01 = vec_comps(image, i, j, P)
            lbp_im[i, j] = lbp_value(vector01)

    lbp_im_crop = lbp_im[R:image.shape[0]-R, R:image.shape[1]-R]

    return lbp_im_crop

# ****************************************************************************


def LBPri_image(image):
    """
    This function computes an image which contains the rotation invariant LBP values of each pixel in which the LBPri is
    calculated

    Parameters
    ----------
    image : numpy array
        Image from which the LBPri values are going to be extracted

    Returns
    -------
    lbp_im_crop : numpy array
        Image with the LBPri values of each pixel of the original image. It is cropped so that only the pixels from
        which the LBPri has been calculated are shown.
    """

    R = 1
    P = 8

    # 1. If it is a colour image, convert it into grayscale
    # ====================== YOUR CODE HERE ======================

    # ============================================================

    # 2. Obtain the limits of the image taking into account the valid points
    x_min, x_max, y_min, y_max = get_limits(image, R)

    # 3. Obtain the LBP value for each valid pixel
    
    # Initialise matrix for the lbp values
    lbp_im = np.zeros(image.shape)

    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            vector01 = vec_comps(image, i, j, P)
            lbp_im[i, j] = lbpri_value(vector01)

    lbp_im_crop = lbp_im[R:image.shape[0]-R, R:image.shape[1]-R]

    return lbp_im_crop

# ****************************************************************************


def LBPriu_image(image):
    """
    This function computes an image which contains the uniform LBP values of each pixel in which the LBPriu is
    calculated

    Parameters
    ----------
    image : numpy array
        Image from which the LBPriu values are going to be extracted

    Returns
    -------
    lbp_im_crop : numpy array
        Image with the LBPriu values of each pixel of the original image. It is cropped so that only the pixels from
        which the LBPriu has been calculated are shown.
    """

    R = 1
    P = 8

    # 1. If it is a colour image, convert it into grayscale
    # ====================== YOUR CODE HERE ======================

    # ============================================================

    # 2. Obtain the limits of the image taking into account the valid points
    x_min, x_max, y_min, y_max = get_limits(image, R)

    # 3. Obtain the LBP value for each valid pixel

    # Initialise matrix for the lbp values
    lbp_im = np.zeros(image.shape)

    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            vector01 = vec_comps(image, i, j, P)
            lbp_im[i, j] = lbpriu_value(vector01)

    lbp_im_crop = lbp_im[R:image.shape[0]-R, R:image.shape[1]-R]

    return lbp_im_crop
