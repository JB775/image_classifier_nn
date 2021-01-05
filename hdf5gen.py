import cv2
import datetime as dt
import h5py
# import matplotlib.pyplot as plt
# import matplotlib.pylab as plb
# import numpy as np
import os
import pandas as pd
from glob import glob



def proc_images():
    """
    Saves compressed, resized images as HDF5 datsets
    Returns
        data.h5, where each dataset is an image or class label
        e.g. X23,y23 = image and corresponding class label
    """
    start = dt.datetime.now()
    # ../input/
    PATH = os.path.abspath(os.path.join('..', 'input'))
    # ../input/sample/images/
    SOURCE_IMAGES = os.path.join(PATH, "test", "elon_small_test")

    # ../input/sample/images/*.png
    # images = glob(os.path.join(SOURCE_IMAGES, "*.png"))
    images = glob(os.path.join(SOURCE_IMAGES, "*.jpg"))

    # Load labels
    labels = pd.read_csv('./input/elon_test_labels1.csv')
       
    # Add a value here and uncomment stuff below if we want to focus on specific labels
    # primaryLabel="Infiltration"
    
    # Size of data
    NUM_IMAGES = len(images)
    HEIGHT = 64
    WIDTH = 64
    CHANNELS = 3
    SHAPE = (HEIGHT, WIDTH, CHANNELS)
    
    with h5py.File('data.h5', 'w') as hf: 
        for i,img in enumerate(images):            
            # Images
            image = cv2.imread(img)
            image = cv2.resize(image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC)
            Xset  = hf.create_dataset(
                name='X'+str(i),
                data=image,
                shape=(HEIGHT, WIDTH, CHANNELS),
                maxshape=(HEIGHT, WIDTH, CHANNELS),
                compression="gzip",
                compression_opts=9)
            # Labels
            base = os.path.basename(img)
            # finding = labels["zzz"][labels["qqq"] == base].values[0]
            yset = hf.create_dataset(
                name='y'+str(i),
                shape=(1,),
                maxshape=(None,),
                compression="gzip",
                compression_opts=9)
            # if primaryLabel in finding:
            #     finding = 1
            #     yset = finding
            # else:
            #     finding = 0
            #     yset = finding
            end=dt.datetime.now()
            print("\r", i, ": ", (end-start).seconds, "seconds", end="")



# TODO: uncomment this if end up using
# proc_images()

