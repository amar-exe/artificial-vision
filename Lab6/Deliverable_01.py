# -*- coding: utf-8 -*-
"""
Created on Sat Apr  25 19:13:48 2016

@author: ealegre
Credits to Victor González from its modifications on 2020.
Reviewed, validated & updated (added comments) by efidalgo (15/03/2022)

Bag of Visual Words:
    0. Read all image names from disk
        - Split in training and test
    1. Obtain features (denseSIFT) for all the images in the training set
    2. Obtaining "visual vocabulary" -> centroid of kmeans features in training
    3. Obtaining histograms of words -> vq
    4. Training SVM
    5. Obtaining descripotors and histograms of words for the test set
    6. Testing SVM
    7. Showing results
    
"""
from os.path import exists, isdir, basename, join, splitext
from os import makedirs
import os
from glob import glob
from random import sample
from scipy import array
from scipy import vstack, amax, amin, ones
from imageio import imread
from scipy.cluster.vq import vq, kmeans2
from scipy.io import loadmat, savemat
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from datetime import datetime
from pickle import dump, load
from cyvlfeat.sift import dsift
from time import perf_counter
from collections import Counter
import pylab as pl
import cv2


IDENTIFIER = '16.04.21'
VERBOSE = True
OVERWRITE = False


class Configuration(object):
    def __init__(self, identifier='amarf'):
        # Include the path to the dataset you just downloaded from Agora
        self.calDir = "/Users/Amar/Downloads/dataset/dataset"
        self.dataDir = 'resultDir'  # should be resultDir or so
        if not exists(self.dataDir):
            makedirs(self.dataDir)
            print("folder " + self.dataDir + " created")
        self.extensions = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
        # Include the number of images per category that the dataset has
        self.imagesperclass = 1000
        # Include the size of the test set
        self.testSize = 0.3
        # Include the number of categories of the dataset
        self.numClasses = 2
        # Include the number of words of the visual vocabulary
        self.numWords = 5
        self.dsiftStep = 5
        self.dsiftSize = 7
        self.dsiftFast = True
        #
        self.descrsPath = join(self.dataDir, identifier + '-descrips.py.mat')
        self.vocabPath = join(self.dataDir, identifier + '-vocab.py.mat')
        self.histPath = join(self.dataDir, identifier + '-hists.py.mat')
        self.histTestPath = join(self.dataDir,
                                 identifier + '-histsTest.py.mat')
        self.modelPath = join(self.dataDir, identifier + '-model.py.mat')
        self.resultPath = join(self.dataDir, identifier + '-result')


def get_classes(datasetpath, numClasses):
    classes_paths = [files
                     for files in glob(datasetpath + "/*")
                     if isdir(files)]
    classes_paths.sort()
    classes = [basename(class_path) for class_path in classes_paths]
    if len(classes) == 0:
        raise ValueError('no classes found')
    if len(classes) < numClasses:
        raise ValueError('conf.numClasses bigger than the number of folders')
    classes = classes[:numClasses]
    return classes


def get_imgfiles(path, extensions):
    all_files = list()
    all_files.extend([join(path, basename(fname))
                     for fname in glob(path + "/*")
                     if splitext(fname)[-1].lower() in extensions])
    return all_files


def get_all_image_nam(classes, conf):
    all_image_nam = list()
    all_image_nam_class_labels = list()
    for i, imageclass in enumerate(classes):
        path = join(conf.calDir, imageclass)
        extensions = conf.extensions
        imgs = get_imgfiles(path, extensions)
        if len(imgs) == 0:
            raise ValueError('no images for class ' + str(imageclass))
        imgs = sample(imgs, conf.imagesperclass)
        all_image_nam = all_image_nam + imgs
        class_labels = list(i * ones(conf.imagesperclass))
        all_image_nam_class_labels = all_image_nam_class_labels + class_labels

    all_image_nam_class_labels = array(all_image_nam_class_labels, 'int')
    return all_image_nam, all_image_nam_class_labels


def is_rgb(image, verbose=False):
    if(len(image.shape) < 3):
        if verbose:
            print('gray')
        return False
    elif len(image.shape) == 3:
        if verbose:
            print('Color(RGB)')
        return True
    else:
        print('others')


def standarizeImage(im):
    if is_rgb(im):
        im = rgb2gray(im)
    im = array(im, 'float32')
    if im.shape[0] > 480:
        # don't remove trailing .0 to avoid integer devision:
        resize_factor = 480.0 / im.shape[0]
        # im = imresize(im, resize_factor)
        im = cv2.resize(im, None, fx=resize_factor, fy=resize_factor)
    if amax(im) > 1.1:
        im = im / 255.0
    assert((amax(im) > 0.01) & (amax(im) <= 1))
    assert((amin(im) >= 0.00))
    return im


def getDescriptors(X_train, conf):
    # selTrainFeats = choice(selTrain, conf.images_for_histogram)
    descriptors = list()
    for imname in X_train:
        im = imread(imname)
        descriptors.append(getDSiftFeatures(im, conf)[1])
    return descriptors

def getDSiftFeatures(imagedata, conf):
    im = standarizeImage(imagedata)
    #Insert the name of the function to compute dense sift descriptors
    frames, descrs = dsift(image=im,
                           step=conf.dsiftStep,
                           size=conf.dsiftSize,
                           fast=conf.dsiftFast)
    return frames, descrs


def get_image_histogram(image_descriptors, vocab, num_words):
    """ Returns the histogram of words for a single image"""
    # For all the descriptors, obtain the closest word using 
    # vector quantization (hint: check scipy)
    code, dist = vq(image_descriptors, vocab)
    # Convert array into Counter of collection
    code_counter = Counter(code)
    # Histogram with the number of words in that image
    hist_image = [code_counter[i] for i in range(num_words)]
    return hist_image


def compute_histograms(descriptors_set, vocab, conf):
    """Returns the histograms of words for each image """
    histograms_set = list()
    for ima_descrips in descriptors_set:
        histograms_set.append(get_image_histogram(ima_descrips,
                                                  vocab, conf.numWords))
    return histograms_set


def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()


#%%
# -------------
# Main Program
# -------------
if __name__ == '__main__':
    conf = Configuration(IDENTIFIER)
    if VERBOSE:
        print('{} finished conf'.format(datetime.now()))

    # --- 0_ Reading data and splitting classes
    # _Obtaining classes and images names
    classes = get_classes(conf.calDir, conf.numClasses)
    # Get names of images
    all_image_nam, all_image_nam_class_labels = get_all_image_nam(classes,
                                                                  conf)
    # _ Splitting the dataset in training and test
    # Splits in training and test. "X" is data "y" are labels
    X_train, X_test, y_train, y_test = train_test_split(
                                    all_image_nam, all_image_nam_class_labels,
                                    test_size=conf.testSize, random_state=0)
    if VERBOSE:
        print('{}: classes and image names read. \
        Dataset splitted'.format(datetime.now()))

# %%# -----------------------
    # 1_ Extracting features
    # -----------------------
    if VERBOSE:
        print('{} start computing descriptors'.format(datetime.now()))
    if (not exists(conf.descrsPath)) | OVERWRITE:
        init_time = perf_counter()
        # Use the custom function created to compute descriptors. Compute them
        # only from the training set you just created.
        desc_train = getDescriptors(y_train, conf)
        # Arrange all descriptors in a single array
        desc_train_stack = vstack(desc_train)
        n_features = desc_train_stack.shape[0]
        savemat(conf.descrsPath, {'descrips': desc_train})
        if VERBOSE:
            # Time expended
            print('Elapsed time in computing descriptors was: \
                  {0:.2f} sec.'.format(perf_counter() - init_time))
    else:
        if VERBOSE:
            print('using old descriptors from {}'.format(conf.descrsPath))
        desc_train = loadmat(conf.descrsPath)['descrips']

# %%# -----------------------
    # 2_ Obtaining vocabulary  - centroids of kmeans
    # -----------------------
    if VERBOSE:
        print('{} start training vocab'.format(datetime.now()))
    if (not exists(conf.vocabPath)) | OVERWRITE:
        init_time = perf_counter()
        # Use kmeans to compute the visual words
        vocab, _ = kmeans2(desc_train_stack.astype(float),
                           k=conf.numWords, minit='points')
        savemat(conf.vocabPath, {'vocab': vocab})
        if VERBOSE:
            print('Elapsed time in computing vocabulary was: \
                  {0:.2f} sec.'.format(perf_counter() - init_time))
    else:
        if VERBOSE:
            print('using old vocab from {}'.format(conf.vocabPath))
        vocab = loadmat(conf.vocabPath)['vocab']

# %%# -----------------------
    # 3_ Obtaining histograms of words as image descriptors for training set
    # -----------------------
    if VERBOSE:
        print('{} start computing hists'.format(datetime.now()))
    if (not exists(conf.histPath)) | OVERWRITE:
        init_time = perf_counter()
        # Use the custom function created to compute histograms.
        histograms_set = compute_histograms(desc_train, vocab, conf)
        savemat(conf.histPath, {'hists': histograms_set})
        if VERBOSE:
            print('Elapsed time in computing histograms was: \
                  {0:.2f} sec.'.format(perf_counter() - init_time))
    else:
        if VERBOSE:
            print('using old hists from {}'.format(conf.histPath))
        histograms_set = loadmat(conf.histPath)['hists']

# %%# -----------------------
    # 4_ Training SVM
    # -----------------------
    if (not exists(conf.modelPath)) | OVERWRITE:
        if VERBOSE:
            print('{} training sklearn svm'.format(datetime.now()))
        # Train an SVM classifier, no arguments for SVC, the default ones
        clf = svm.SVC()
        if VERBOSE:
            print(clf)
        clf.fit(histograms_set, y_train)
        with open(conf.modelPath, 'wb') as fp:
            dump(clf, fp)
    else:
        if VERBOSE:
            print('loading old SVM model')
        with open(conf.modelPath, 'rb') as fp:
            clf = load(fp)

# %%# -----------------------
    # 5_ Obtaining descriptors and histograms of words for the test set
    # -----------------------
    # 5.1_ Descriptors for the test set
    desc_test = getDescriptors(X_test, conf)

    # 5.2_ Histograms of words for the test set
    if VERBOSE:
        print('{} start computing test histograms'.format(datetime.now()))
    if (not exists(conf.histTestPath)) | OVERWRITE:
        init_time = perf_counter()
        hists_test = compute_histograms(desc_test, vocab, conf)
        savemat(conf.histTestPath, {'histsTest': hists_test})
        if VERBOSE:
            print('Elapsed time in computing histograms was: \
                  {0:.2f} sec.'.format(perf_counter() - init_time))
    else:
        if VERBOSE:
            print('using old hists from {}'.format(conf.histPath))
        hists_test = loadmat(conf.histTestPath)['histsTest']

# %%# -----------------------
    # 6_ Testing SVM
    # -----------------------
    if (not exists(conf.resultPath)) | OVERWRITE:
        if VERBOSE:
            print('{} testing svm'.format(datetime.now()))
        # Make the prediction using the testing set
        predicted_classes = clf.predict(hists_test)
        true_classes = y_test
        accuracy = accuracy_score(true_classes, predicted_classes)
        # Compute the confussion matrix with the true and predicted classes
        cm = confusion_matrix(y_true, y_predicted)
        with open(conf.resultPath, 'wb') as fp:
            dump(conf, fp)
            dump(cm, fp)
            dump(predicted_classes, fp)
            dump(true_classes, fp)
            dump(accuracy, fp)
    else:
        with open(conf.resultPath, 'rb') as fp:
            conf = load(fp)
            cm = load(fp)
            predicted_classes = load(fp)
            true_classes = load(fp)
            accuracy = load(fp)
# %%# -----------------------
    # 7_ Showing Results
    # -----------------------
    print("accuracy = {0:.4f}".format(accuracy))
    # Print and show in the Plot tab the confusion matrix previously created.
    print(cm)
    showconfusionmatrix(cm)
