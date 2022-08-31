# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 09:19:24 2016
Updated on Apr 14 2018
Updated on Apr 28 2021

@author: ealegre
"""

# %% check_path: Check if a folder exists


def check_path(name_folder, toread=True, verbose=False):
    """
    check_path(name_folder,toread=False,verbose=False)
    INPUT:
        1 name_folder: name of the directory to check
        2 toread: boolean. Has to be true to read from the specifed folder
        (name_folder)
                When is true, if the folder does not exist, it is created
        3 verbose: You know.
    OUTPUT:

    Kk_April_2016
    """

# Check if the folder exists, so that it is correctly created if not

    import os
    import errno

    if toread:
        if os.path.isdir(name_folder):
            if verbose:
                print('\nIt exists and it is a directory')
            path_OK = True
            return path_OK
        else:
            print('\nIt is not a directory!')
            path_OK = False
            return path_OK

    else:  # No read but Write. Create the directory
        try:
            os.makedirs(name_folder)
            if verbose:
                print('\nDirectory {} successfully created!'
                      .format(name_folder))
            path_OK = True
            return path_OK
        except OSError as exception:
            if exception.errno != errno.EEXIST or not os.path.isdir(name_folder):
                raise
            if verbose:
                print('\nIt does NOT exist or is not a directory')
            path_OK = False
            return path_OK
        else:
            print('\nBE CAREFUL! Directory {} already exists.'
                  .format(name_folder))
            path_OK = True
            return path_OK


# %% read_image_names


def read_image_names(path_ima='./images', img_ext='*.png', verbose=False):
    """
    read_image_names(path_ima='./images', img_ext='*.png',verbose=False)

    Given a path and an image file extension, this function introduce all
    the image name contained in the folder passed in the path in a list and
    it returns it. The introduced name will contain the relative path
    (the directory concatenate with the image name).

    INPUT:
        path_imag: string containing the absolute or relative path.
            By default path_imag= './images'
        img_ext: string containing the extension of the image files to be read.
            By default img_ext='*.png'
        verbose: boolean. If True, messages are shown.
            By default verbose=False

    OUTPUT:
        full_names_img: list containing file names with its path (in a sigle
        string).

    Kk.April.2016
    """
    import os
    import glob

    # List to be returned
    full_names_img = list()

    # % % 1 Print the original path and check if the folder with images exists!
    prev_path = os.getcwd()
    if verbose:
        print('(1) Original path was:  {} \n'.format(prev_path))

    if os.path.isdir(path_ima):
        if verbose:
            print('(2) The directory {} does exists!\n'.format(path_ima))
    else:
        if verbose:
            print('(2) WARNING: There is not such directory\n')

    # % % 2_ Read all the files with the specified "extension" in the current
    # working directory
    full_names_img = glob.glob(os.path.join(path_ima, img_ext))
    # glob.glob(os.path.join(path_img, img_ext))

    if verbose:
        print('(3) The "{0}" directory has {1} images with {2} extension.\n'
              .format(os.getcwd(), len(full_names_img), img_ext))

    return full_names_img


# %% get_labels_from_names


def get_labels_from_names(list_names, lab_class1, lab_class2):
    """
    get_labels_from_names(list_names, lab_class1, lab_class2)

    To use it:
    my_labels,num_ele_class1,num_ele_class2 =
                    get_labels_from_names (list_names,'weapon','tool')

    Kk.April.2016
    """
    list_labels = [lab_class1 if lab_class1 in f.split('/')[-1] else lab_class2
                   for f in list_names]

    num_ele_class1 = list_labels.count('weapon')
    num_ele_class2 = list_labels.count('tool')

    return list_labels, num_ele_class1, num_ele_class2


# %% average_image_shape: CREATE A FUNCTION THAT RETURNS THE AVERAGE SHAPE OF
# ALL THE IMAGES IN A FOLDER


def average_image_shape(full_names_img):
    """
    average_image_shape(full_names_img)
    Takes a list containing the full names (whole path, at least relative
    path) of images.
    Returns a two elements tuple with the average number of
    rows and columns

    INPUT:
        full_names_image: list with full names of images

    OUTPUT:
        average_shape_images: (averagerows, averagecolums)
         type(average_shape_images) -> tuple

    Kk.April.2016
    """
    from scipy import misc

    # Initialize variables
    numrows = list()
    numcols = list()
    average_shape_images = ()  # empty tuple

    # Read sizes and introduce them in lists
    for nameimage in full_names_img:
        image = misc.imread(nameimage)
        rows, cols, _ = image.shape
        numrows.append(rows)
        numcols.append(cols)

    # Compute the average
    average_shape_images = (round(sum(numrows)/len(numrows)),
                            round(sum(numcols)/len(numcols)))
    return average_shape_images

# %% down_one_page_of_images_color


def down_one_page_of_images_color(query, image_type, directory='images'):
    """
       down_one_page_of_images(query, image_type, directory='images'):
       Download one page of images from BING search engine, given as argument
       the query, the image_type and the directory where they will be saved.

       INPUT:
           query = String with the query term. Example: "weapons"
           image_type = String with the image_type term. Example: "weapons"
           directory='images'
    Kk.2016
    """
    from bs4 import BeautifulSoup
    import requests
    import re
    import urllib.request
    import os
    import time

    # Function to obtain the "Soup" from the URL selected
    def get_soup(url):
        return BeautifulSoup(requests.get(url).text, "lxml")

    # Using BING url with the query term specifyed (input parameter)
    url = "http://www.bing.com/images/search?q=" + query + \
        "&qft=+filterui:color2-color+filterui:imagesize-large&FORM=R5IR3"

    # Getting the soup
    soup = get_soup(url)
    # And getting the images names
    images = [a['src'] for a in soup.find_all("img",
              {"src": re.compile("mm.bing.net")})]

    # Retrieving the images and writing them to disk
    secods_waiting = (5, 3, 4)
    ind_sec = 0
    for img in images:
        raw_img = urllib.request.urlopen(img).read()
        cntr = len([i for i in os.listdir(directory) if image_type in i]) + 1
        f = open("{0}/{1}_{2}.png".format(directory, image_type, cntr), 'wb')
        f.write(raw_img)
        f.close()

        # Trying to cheat to BING
        if ind_sec == 0:  # Wait because BING  will return "time out" otherwise
            time.sleep(secods_waiting[ind_sec])
            ind_sec = ind_sec + 1
        elif ind_sec == 1:
            time.sleep(secods_waiting[ind_sec])
            ind_sec = ind_sec + 1
        else:
            time.sleep(secods_waiting[ind_sec])
            ind_sec = 0
# %% down_one_page_of_images


def down_one_page_of_images(query, image_type, directory='images'):
    """
       down_one_page_of_images(query, image_type, directory='images'):
       Download one page of images from BING search engine, given as argument
       the query, the image_type and the directory where they will be saved.

       INPUT:
           query = String with the query term. Example: "weapons"
           image_type = String with the image_type term. Example: "weapons"
           directory='images'
    Kk.2016
    """
    from bs4 import BeautifulSoup
    import requests
    import re
    import urllib.request
    import os
    import time

    # Function to obtain the "Soup" from the URL selected
    def get_soup(url):
        return BeautifulSoup(requests.get(url).text, "lxml")

    # Using BING url with the query term specifyed (input parameter)
    url = "http://www.bing.com/images/search?q=" + query + \
        "&qft=+filterui:color2-bw+filterui:imagesize-large&FORM=R5IR3"

    # Getting the soup
    soup = get_soup(url)
    # And getting the images names
    images = [a['src'] for a in soup.find_all("img",
              {"src": re.compile("mm.bing.net")})]

    # Retrieving the images and writing them to disk
    secods_waiting = (5, 3, 4)
    ind_sec = 0
    for img in images:
        raw_img = urllib.request.urlopen(img).read()
        cntr = len([i for i in os.listdir(directory) if image_type in i]) + 1
        f = open("{0}/{1}_{2}.png".format(directory, image_type, cntr), 'wb')
        f.write(raw_img)
        f.close()

        # Trying to cheat to BING
        if ind_sec == 0:  # Wait because BING  will return "time out" otherwise
            time.sleep(secods_waiting[ind_sec])
            ind_sec = ind_sec + 1
        elif ind_sec == 1:
            time.sleep(secods_waiting[ind_sec])
            ind_sec = ind_sec + 1
        else:
            time.sleep(secods_waiting[ind_sec])
            ind_sec = 0


# %% plot_img_and_hist
def plot_img_and_hist(img, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    from matplotlib import pyplot as plt
    import skimage as ski

    img = ski.img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    ax_img.set_adjustable('box-forced')

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = ski.exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf
