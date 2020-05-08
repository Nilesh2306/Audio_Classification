The challenge was hosted on the Kaggle platform [1] as “FreesoundGeneral-PurposeAudioTaggingChallenge”. The goal of the task was to build an audio tagging system that can recognize the category of an audio clip from a subset of 41 diverse categories drawn from the Audio Set Ontology. The baseline model[2] uses the shallow Convolutional neural networks which is a scaled down version of general Deep network used in computer vision domain.They used the log mel spectrogram and three convolutional neural networks with 100,150 and 300 channels to classify audio labels. This model produced MAP score of around 70.





## To run the project, extract the zip containing all project files into 'project' directory and change the path of that directory at the beginning of main notebook.

## Run the task 1 to download and prepare dataset or else download and unzip the dataset into '.project/dataset' directory.


References:
[1] https://www.kaggle.com/c/freesound-audio-tagging/data
[2] Eduardo Fonseca, Manoj Plakal, Frederic Font, Daniel P. W. Ellis, Xavier Favory, Jordi Pons, and Xavier Serra. General-purpose tagging of freesound audio with audioset labels: task description, dataset, and baseline. Submitted to DCASE2018 Workshop, 2018. URL: https://arxiv.org/abs/1807.09902, arXiv:1807.09902
