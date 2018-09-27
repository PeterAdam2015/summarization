import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from utils.create_vocab import load_vocab
from models import baseline
from torch.utils.data import DataLoader
from utils.create_datasets import SumDatasets
import utils.config as config

from models import baseline


def encoder_transform(features_1, features_2, features_3, features_4):
    """
    To transform the data to the format, we need to order the data in the sequences of length
    ,so we can later utilize the data in the NN model with the pack_paded_sequences and pad_packed_sequences.
    
    Parameters
    ----------
        
    """
    assert isinstance(features_1, torch.Tensor), "You must give the data to tensor object"
    batch_size = features_1.size(0)
    if batch_size == 1:
        pass
    else:
        sorted_length, sorted_idx = features_2.sort()  # sort will return both the ascending sorted value and also the sorted index
        reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()  # this will contain the batch_size-1
        sorted_length, sorted_idx = sorted_length[reverse_idx], sorted_idx[reverse_idx]
        features_1 = features_1[sorted_idx]
        features_2 = features_2[sorted_idx]
        features_3 = features_3[sorted_idx]
        features_4 = features_4[sorted_idx]
    features_1.squeeze_(1), features_3.squeeze_(1), features_4.squeeze_(1)
    features_3 = features_3.permute([1, 0])
    features_4 = features_4.permute([1, 0])
    return features_1, features_2, features_3, features_4

class Train(object):
    """
    train class for encoder-decoder architure

    """

    def __init__(self):
        self.vocab = load_vocab(config.vocab_path)
        self.train_path = config.train_path
        self.data_loader = DataLoader(SumDatasets(self.train_path), config.batch_size, shuffle=True)
        self.criterion = nn.NLLLoss()
        self.epoches = config.epoches
        # TODO, add tnesorflow tensorboard to visualzie the training process



    def setup_train(self, model_path = None, optimizer=None):
        self.model = baseline.BaseModel(model_file_path=model_path)
        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        initial_lr = config.lr
        if optimizer is None:
            # can we directly use the whole parameters for training?
            self.optimizer = torch.optim.SGD(params, lr=initial_lr)
        start_iter, start_loss = 0, 0

        # load the existing model if it exist:
        if model_path is not None:
            state = torch.load(model_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

        return start_iter, start_loss
    

    def train_batch(self, features):
        # to clear the gradients before training this batch
        self.optimizer.zero_grad()
        features_1, features_2, features_3, features_4 = encoder_transform(*features)
        outputs, hidden = self.model.encoder(features_1, features_2)
        loss = 0
        for di in range(features_3.size(0)):
            output, hidden = self.model.decoder(features_3[di], hidden)
            loss += self.criterion(output, features_4[di])
        # TODO, mask the loss in the target, to avoid the uncesssary loss computa
        loss.backward()
        self.optimizer.step()
        return loss

    def train_epoches(self):
        """
        an epoch trianing for the training data, to loop over the entire datasets
        
        """
        
        for epoch in range(1, self.epoches + 1):
            # train bathes in this epoch
            start_iter, epoch_loss = self.setup_train()
            print_loss_total = 0  # Reset every print_every
            for di, features in enumerate(tqdm(self.data_loader)):
                batch_loss = self.train_batch(features)
                print_loss_total += batch_loss
                epoch_loss += batch_loss
                if (di+1) % config.print_every == 0:
                    print_loss_avg = print_loss_total / config.print_every / config.batch_size
                    print_loss_total = 0
                    print('%d have %.4f' % (di, print_loss_avg))
            print("epoch {} : Loss:{}".format(epoch, epoch_loss/len(self.data_loader)/config.batch_size))
            
