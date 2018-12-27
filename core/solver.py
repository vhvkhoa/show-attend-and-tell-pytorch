import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os
from scipy import ndimage
from scipy.misc import imresize
from tqdm import tqdm
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from .utils import *
from .dataset import CocoCaptionDataset

def pack_collate_fn(batch):
    features, cap_vecs, captions = zip(*batch)

    len_sorted_idx = sorted(range(len(cap_vecs)), key=lambda x: len(cap_vecs[x]), reverse=True)
    len_sorted_cap_vecs = [np.array(cap_vecs[i]) for i in len_sorted_idx]
    len_sorted_features = torch.tensor([features[i] for i in len_sorted_idx])
    len_sorted_captions = [captions[i] for i in len_sorted_idx]
    seq_lens = torch.tensor([len(cap_vec) for cap_vec in len_sorted_cap_vecs])

    packed_cap_vecs = nn.utils.rnn.pack_sequence([torch.from_numpy(cap_vec) for cap_vec in len_sorted_cap_vecs])

    return len_sorted_features, packed_cap_vecs, len_sorted_captions, seq_lens

class CaptioningSolver(object):
    def __init__(self, model, word_to_idx, train_dataset, val_dataset, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - snapshot_steps: Integer; training losses will be printed every snapshot_steps iterations.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_checkpoint: String; model path for test
        """

        self.model = model
        self.word_to_idx = word_to_idx
        self._start = word_to_idx['<START>']
        self._null = word_to_idx['<NULL>']
        self._end = word_to_idx['<END>']

        self.n_time_steps = kwargs.pop('n_time_steps', 31)
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('optimizer', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.metric = kwargs.pop('metric', 'CIDEr')
        self.alpha_c = kwargs.pop('alpha_c', 1.0)
        self.snapshot_steps = kwargs.pop('snapshot_steps', 100)
        self.eval_every = kwargs.pop('eval_every', 200)
        self.start_from = kwargs.pop('start_from', 0)
        self.log_path = kwargs.pop('log_path', './log/')
        self.checkpoint_dir = kwargs.pop('checkpoint_dir', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', '')
        self.test_checkpoint = kwargs.pop('test_checkpoint', './model/lstm/model-1')

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=pack_collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=pack_collate_fn)

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        elif self.update_rule == 'rmsprop':
            self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=self.learning_rate, momentum=0.9)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self._null)

        self.train_engine = Engine(self._train)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
    
    def _train(self, engine, batch):
        features, packed_cap_vecs, captions, seq_lens = batch
        self.optimizer.zero_grad()

        cap_vecs, batch_sizes = packed_cap_vecs
        features = self.model.batch_norm(features)
        features_proj = self.model.project_features(features)
        hidden_states, cell_states = self.model.get_initial_lstm(features)
        inv_seq_lens = 1 / seq_lens

        loss = 0
        alphas = []

        start_idx = 0
        for i in range(len(batch_sizes)-1):
            end_idx = start_idx + batch_sizes[i]
            curr_cap_vecs = cap_vecs[start_idx:end_idx]
            print(curr_cap_vecs.size())
            logits, alpha, (hidden_states, cell_states) = self.model(features[:batch_sizes[i]],
                                                                     features_proj[:batch_sizes[i]],
                                                                     curr_cap_vecs,
                                                                     hidden_states[:, :batch_sizes[i]],
                                                                     cell_states[:, :batch_sizes[i]])
            loss += self.criterion(logits[:batch_sizes[i+1]], cap_vecs[end_idx:end_idx+batch_sizes[i+1]])
            if self.alpha_c > 0:
                alpha_reg = self.alpha_c * torch.sum((seq_lens[:batch_sizes[i+1]] - alpha) ** 2)
                loss += alpha_reg

            alphas.append(alpha)
            start_idx = end_idx
        
        loss.backward()
        self.optimizer.step()
    
    def train(self):
        self.train_engine.run(self.train_loader)
