import torch
import torch.nn as nn
import torch.nn.functional as F

class Evaluation(object):
    """evaluation the model accurding to the local test data.
    """

    def __init__(self, model_path, test_data):
        self.model_path = model_path
        self.test_data = test_data

        
