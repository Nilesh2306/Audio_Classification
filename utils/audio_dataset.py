"""
This module provides the necessary functions required to prepare the audio dataset.
Dataset will be downloaded from the "https://zenodo.org/record/2552860#.XpvprshKjIV" and all files will be unzipped in
the dataset directory.
"""

import wget
import os
import zipfile
import shutil


def unzip_data():
    for file in os.listdir("dataset"):
        if file.endswith(".zip"):
            zip_ref = zipfile.ZipFile('dataset/' + file, 'r')
            zip_ref.extractall('dataset/')  # extract the data
            print("Data in the file %s is extracted successfully." % file)
            zip_ref.close()
            os.remove('dataset/' + file)  # remove the zip file after unzipping


def prepare_audio_dataset():
    # create directory to store dataset if not already exist.
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    if os.path.exists('dataset'):
        shutil.rmtree('dataset')  # delete all previous files present in the dataset
        os.mkdir('dataset')

    # download all files from "https://zenodo.org/record/2552860#.XpvprshKjIV"
    print("Downloading training data...")
    wget.download('https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_train.zip?download=1', out='dataset')
    print("Downloading test data...")
    wget.download('https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_test.zip?download=1', out='dataset')
    print("Downloading document data...")
    wget.download('https://zenodo.org/record/2552860/files/FSDKaggle2018.doc.zip?download=1', out='dataset')
    print("Downloading meta data...")
    wget.download('https://zenodo.org/record/2552860/files/FSDKaggle2018.meta.zip?download=1', out='dataset')
    print("All files are downloaded from the source path. Starting to unzip the data...")
    # call the function to unzip data files
    unzip_data()
