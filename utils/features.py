"""
This module provides the necessary functions required to extract the features from audio files.
Part of the feature extractions(random offset & padding) is part of the kaggle
kernel notebook: https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data/comments
"""

import pandas as pd
import librosa
import os
import numpy as np
import random
random.seed(111)

train = pd.read_csv('dataset/FSDKaggle2018.meta/train_post_competition.csv')  # read meta data
test = pd.read_csv('dataset/FSDKaggle2018.meta/test_post_competition_scoring_clips.csv')


def extract_mfcc(audio_data, sampling_rate, audio_duration):
    input_length = audio_duration * sampling_rate
    audio_trimmed = librosa.effects.trim(audio_data)[0]  # trim the silent part
    # Random offset or Padding
    if len(audio_trimmed) > input_length:
        max_offset = len(audio_trimmed) - input_length
        offset = np.random.randint(max_offset)
        audio_trimmed = audio_trimmed[offset:(input_length + offset)]
    else:
        if input_length > len(audio_trimmed):
            max_offset = input_length - len(audio_trimmed)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        audio_trimmed = np.pad(audio_trimmed, (offset, input_length - len(audio_trimmed) - offset), "constant")
    mfcc = librosa.feature.mfcc(y=audio_trimmed, sr=sampling_rate, n_mfcc=40)
    return mfcc


def extract_logmel(audio_data, sampling_rate=44100, audio_duration=2):
    n_mels = 64
    hop_length = 512
    n_fft = 2048
    input_length = audio_duration * sampling_rate
    audio_trimmed = librosa.effects.trim(audio_data)[0]  # trim the silent part
    # Random offset / Padding
    if len(audio_trimmed) > input_length:
        max_offset = len(audio_trimmed) - input_length
        offset = np.random.randint(max_offset)
        audio_trimmed = audio_trimmed[offset:(input_length + offset)]
    else:
        if input_length > len(audio_trimmed):
            max_offset = input_length - len(audio_trimmed)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        audio_trimmed = np.pad(audio_trimmed, (offset, input_length - len(audio_trimmed) - offset), "constant")
    b = np.abs(librosa.stft(audio_trimmed[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))
    b = np.abs(librosa.stft(audio_trimmed, n_fft=n_fft, hop_length=hop_length))
    b_d = librosa.amplitude_to_db(b, ref=np.max)
    mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels)
    L = librosa.feature.melspectrogram(audio_trimmed, sr=sampling_rate, n_fft=n_fft, hop_length=hop_length,
                                       n_mels=n_mels)
    logmel = librosa.power_to_db(L, ref=np.max)
    return logmel


""" Inspired by : https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6 """


def shift_audio(data, sampling_rate=44100, shift_max=2):
    shift = np.random.randint(sampling_rate * shift_max)
    shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data


""" This function is a part of kaggle notebook "https://www.kaggle.com/davids1992/specaugment-quick-implementation".
This is implementation of the augmentation method described in the paper: https://arxiv.org/abs/1904.08779    
"""


def spec_augment(spec: np.ndarray, num_mask=2,
                 freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)

        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0

    return spec


def generate_features(feature_type, sampling_rate=44100, audio_duration=2):
    """
    :param feature_type: Takes two values 'mfcc' or 'logmel' to indicate type of feature to be extracted
    :param sampling_rate: Sampling rate which should be used for audio processing. Default:44100
    :param audio_duration: Audio duration for the pre-processing. All the audio files will be pre processed to same length.
    :return: Returns two lists containing train and test features
     """
    train_features = []
    test_features = []
    print("Extracting %s features of training data" % feature_type)
    for i in train.index:
        fname = train['fname'][i]
        audio_data, _ = librosa.load('dataset/FSDKaggle2018.audio_train/' + fname, sr=sampling_rate,
                                     res_type="kaiser_fast")
        if feature_type == 'mfcc':
            train_features.append(extract_mfcc(audio_data, sampling_rate, audio_duration))
        elif feature_type == 'logmel':
            train_features.append(extract_logmel(audio_data, sampling_rate, audio_duration))
        else:
            raise ValueError('unknown feature name: %s' % feature_type)
            break
        if i % 100 == 0:
            print('Features extracted for %d files.' % i)
    print("Extracting %s features of test data" % feature_type)
    for i in test.index:
        fname = test['fname'][i]
        audio_data, _ = librosa.load('dataset/FSDKaggle2018.audio_test/' + fname, sr=sampling_rate,
                                     res_type="kaiser_fast")
        if feature_type == 'mfcc':
            test_features.append(extract_mfcc(audio_data, sampling_rate, audio_duration))
        elif feature_type == 'logmel':
            test_features.append(extract_logmel(audio_data, sampling_rate, audio_duration))
        else:
            raise ValueError('unknown features: %s' % feature_type)
            break
        if i % 100 == 0:
            print('Features extracted for %d files.' % i)
    return train_features, test_features


def generate_augmented_features(feature_type, sampling_rate=44100, audio_duration=2):
    """
    :param feature_type: Takes two values 'mfcc' or 'logmel' to indicate type of feature to be extracted. Different data
                        augmentation will be used for logmel & mfcc.
    :param sampling_rate: Sampling rate which should be used for audio processing. Default:44100
    :param audio_duration: Audio duration for the pre-processing. All the audio files will be pre processed to same length.
    :return: Returns two lists containing train augmented features and train labels.
     """
    train_aug_features = []
    train_aug_labels = []
    # minor_class = train['label'].value_counts()[-15:].index.tolist() # define list if minor labels
    print("Extracting augmented %s features of training data" % feature_type)
    for i in train.index:
        fname = train['fname'][i]
        label = train['label'][i]
        audio_data, _ = librosa.load('dataset/FSDKaggle2018.audio_train/' + fname, sr=sampling_rate,
                                     res_type="kaiser_fast")
        if feature_type == 'mfcc':
            augmented_data = shift_audio(audio_data)
            train_aug_features.append(extract_mfcc(audio_data, sampling_rate, audio_duration))
            train_aug_labels.append(label)
            train_aug_features.append(extract_mfcc(augmented_data, sampling_rate, audio_duration))
            train_aug_labels.append(label)
        elif feature_type == 'logmel':
            x = extract_logmel(audio_data, sampling_rate, audio_duration)
            train_aug_features.append(x)
            train_aug_labels.append(label)
            train_aug_features.append(spec_augment(x))
            train_aug_labels.append(label)
        else:
            raise ValueError('unknown feature name: %s' % feature_type)
            break
        if i % 100 == 0:
            print('Features extracted for %d files.' % i)
            
    return train_aug_features, train_aug_labels
