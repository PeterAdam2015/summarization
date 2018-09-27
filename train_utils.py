import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.create_vocab import load_vocab
from models import baseline
from torch.utils.data import DataLoader
from utils.create_datasets import SumDatasets
import utils.config as config

from models import baseline
class Train(object):
    """
    train class for encoder-decoder architure

    """

    def __init__(self):
        self.vocab = load_vocab(config.vocab_path)
        self.train_path = config.train_path
        self.data_loader = DataLoader(SumDatasets(self.train_path), config.batch_size, shuffle=True)
        # TODO, add tnesorflow tensorboard to visualzie the training process



    def setup_train(self, model_path = None, optimizer=None):
        self.model = baseline.BaseModel(model_file_path=model_path)
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        initial_lr = config.lr
        if optimizer is None:
            self.optimizer = torch.optim.SGD(params, lr=initial_lr)
        start_iter, start_loss = 0, 0

        # load the existing model if it exist:
        if model_path is not None:
            state = torch.load(model_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']


        return start_iter, start_loss

    def train_batch(self):
        enc_batch, enc_lens, dec_batch, targets = next(self.data_loader)

