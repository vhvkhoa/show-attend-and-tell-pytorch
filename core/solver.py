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
from .beam_decoder import BeamSearchDecoder 

def pack_collate_fn(batch):
    features, cap_vecs, captions = zip(*batch)

    len_sorted_idx = sorted(range(len(cap_vecs)), key=lambda x: len(cap_vecs[x]), reverse=True)
    len_sorted_cap_vecs = [np.array(cap_vecs[i]) for i in len_sorted_idx]
    len_sorted_features = torch.tensor([features[i] for i in len_sorted_idx])
    len_sorted_captions = [captions[i] for i in len_sorted_idx]
    seq_lens = torch.tensor([len(cap_vec) for cap_vec in len_sorted_cap_vecs], dtype=torch.float)

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
        self.batch_size = kwargs.pop('batch_size', 100)
        self.beam_size = kwargs.pop('beam_size', 3)
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
        self.device = kwargs.pop('device', 'cuda:1')

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=pack_collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=4)

        self.beam_decoder = BeamSearchDecoder(self.model, self.beam_size, len(self.word_to_idx), self._start, self._end, self.n_time_steps)

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        elif self.update_rule == 'rmsprop':
            self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=self.learning_rate, momentum=0.9)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self._null)

        self.train_engine = Engine(self._train)
        self.test_engine = Engine(self._test)

        self.train_engine.add_event_handler(Events.ITERATION_COMPLETED, self.training_end_iter_handler)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
    
    def training_end_iter_handler(self, engine):
        iteration = engine.state.iteration
        epoch = engine.state.epoch
        loss = engine.state.output

        print(loss)
        if (iteration + 1) % self.snapshot_steps == 0:
            print('Epoch: {}, Iteration:{}, Loss:{}'.format(epoch, iteration + 1, loss))
            if (iteration + 1) % self.eval_every == 0:
                self.test(self.val_loader, is_validation=True)
    
    def testing_end_epoch_handler(self, engine, is_val):
        captions = engine.state.output
        save_json(captions, './data/%s/%s.candidate.captions.json')

        if is_val: 
            caption_scores = evaluate(get_scores=True)
            write_scores(caption_scores, './', engine.state.epoch, engine.state.iteration)
        
    def _train(self, engine, batch):
        features, packed_cap_vecs, captions, seq_lens = batch
        features = features.to(device=self.device)
        seq_lens = seq_lens.to(device=self.device)

        cap_vecs, batch_sizes = packed_cap_vecs
        cap_vecs = cap_vecs.to(device=self.device)
        batch_sizes = batch_sizes.to(device=self.device)
        features = self.model.batch_norm(features)
        features_proj = self.model.project_features(features)
        hidden_states, cell_states = self.model.get_initial_lstm(features)

        total_loss = 0
        alphas = []

        start_idx = 0
        for i in range(len(batch_sizes)-1):
            self.optimizer.zero_grad()
            end_idx = start_idx + batch_sizes[i]
            curr_cap_vecs = cap_vecs[start_idx:end_idx]

            logits, alpha, (hidden_states, cell_states) = self.model(features[:batch_sizes[i]],
                                                                     features_proj[:batch_sizes[i]],
                                                                     curr_cap_vecs,
                                                                     hidden_states[:, :batch_sizes[i]],
                                                                     cell_states[:, :batch_sizes[i]])
            loss = self.criterion(logits[:batch_sizes[i+1]], cap_vecs[end_idx:end_idx+batch_sizes[i+1]])
            total_loss += loss.item() 
            loss.backward(retain_graph=True)
            self.optimizer.step()

            alphas.append(alpha)
            start_idx = end_idx
        
        if self.alpha_c > 0:
            alphas = nn.utils.rnn.pad_sequence(alphas)
            alphas_reg = self.alpha_c * torch.sum((torch.unsqueeze(seq_lens, -1) - torch.sum(alphas, 1)) ** 2)
        
        return total_loss
    
    def _test(self, engine, batch_features):
        cap_vecs = self.beam_decoder.decode(batch_features)
        cap_vec = cap_vecs.data.cpu().numpy()
        return decode_captions(cap_vecs, self.word_to_idx)

    def train(self, num_epochs=10):
        self.model.train()
        print(self.model)
        self.train_engine.run(self.train_loader, max_epochs=num_epochs)

    def test(self, test_dataset=None, is_validation=False):
        self.test_engine.add_event_handler(Events.EPOCH_COMPLETED, self.testing_end_epoch_handler, is_validation)    
        self.model.eval()

        if test_dataset == None:
            self.test_engine.run(self.val_loader)
        else:
            self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=4, collate_fn=pack_collate_fn)
            self.test_engine.run(self.test_loader)
        