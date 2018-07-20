import os

import wget

from .preprocess_data import generate_data

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 18

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 24

# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 12

# Batch Size
BATCH_SIZE = 100

# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128


def train(dataset = 'Opportunity'):
    if dataset is not 'Opportunity':
        raise NotImplementedError()

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset" \
          ".zip"
    Opportunity_UCIDataset_zip_dir = wget.download(url)

    generate_data(Opportunity_UCIDataset_zip_dir,
                  os.path.join("dataset", dataset + ".pckl")
                  )
