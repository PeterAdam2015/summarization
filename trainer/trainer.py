from __future__ import division
import logging
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import config




class SupervisedTrainer(object):
    r"""
    an universal api for model in the models folder
    the trainier will accept the loss citeration and define the batch_size the
    print_every of the dev training sets loss and also the checkpoint it saved 
    also print the dev information in the dev datasets, including the rouge - l metric
    """
    def __init__(self, expt_name="simple train", expt_dir="experiment", loss=None, batch_size=64,
                random_seed=None, device=-1, checkpoint_every=100, print_every=100,
                show_dev_information=True):
            self.expt_name = expt_name
            self.random_seed = random_seed
            if not random_seed:
                random.seed(seed)
                torch.manual_seed(seed)
            self.expt_dir = expt_dir
            if not os.path.exists(self.expt_dir):
                # to store the check_point
                os.mkdir(self.expt_dir)
            if loss is None:
                print("must given the default loss, else, NLLoss will be applied!")
                loss = nn.NLLLoss()
            self.loss = loss
            if optimizer is None:
                optimizer = torch.optim.Adam
            self.optimizer = optimizer
            self.device = device
            self.checkpoint_every = checkpoint_every
            self.print_every = print_every
            self.show_dev_information
    
    def train_batch(self, input_variables, input_length, target_variables, model,
                    teaching_forcing=0.5):
        """implement both the seq2seq model for supervised leraning and 

        
        Parameters
        ----------
        input_variables : tensor with dynamic length
            [description]
        target_variables : [type]
            [description]
        model : torch.nn.Module
            the model used for training
        input_length : [type], optional
            [description] (the default is None, which descirbe the input variable length)
        teaching_forcing : float, optional
            [description] (the default is 0.5, which decide the probablity of use target varible as input)
        
        """
        loss = self.loss
        decoder_outputs, decoder_hidden, other = model(input_variables, input_length, target_variables,
                                                        teaching_forcing_ratio=teaching_forcing)
        acc_loss = 0
        loss.zero_grad()
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variables.size(0)
            acc_loss += loss.eval(decoder_outputs, target_variables)
            