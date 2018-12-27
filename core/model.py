# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

class CaptionGenerator(nn.Module):
    def __init__(self, feature_dim=[196, 512], embed_dim=512, hidden_dim=1024,
                  prev2out=True, ctx2out=True, enable_selector=True, dropout=0.5, len_vocab=10000):
        super(CaptionGenerator, self).__init__()
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.enable_selector = enable_selector
        self.dropout = dropout
        self.V = len_vocab
        self.L = feature_dim[0]
        self.D = feature_dim[1]
        self.M = embed_dim
        self.H = hidden_dim 
        
        # Trainable parameters :
        self.lstm_cell = nn.LSTM(self.D + self.M, self.H, dropout=0.5)
        self.hidden_state_init_layer = nn.Linear(self.D, self.H)
        self.cell_state_init_layer = nn.Linear(self.D, self.H)
        self.embedding_lookup = nn.Embedding(self.V, self.M)
        self.feats_proj_layer = nn.Linear(self.D, self.D)
        self.hidden_to_attention_layer = nn.Linear(self.H, self.D) 
        self.attention_layer = nn.Linear(self.D, 1)
        self.selector_layer = nn.Linear(self.H, 1)
        self.hidden_to_embedding_layer = nn.Linear(self.H, self.M)
        self.context_to_embedding_layer = nn.Linear(self.D, self.M)
        self.embedding_to_output_layer = nn.Linear(self.M, self.V)

        # functional layers
        self.features_batch_norm = nn.BatchNorm1d(self.L)
        self.dropout = nn.Dropout(p=dropout)

    def get_initial_lstm(self, features):
        features_mean = torch.mean(features, 1)
        h = torch.tanh(self.hidden_state_init_layer(features_mean)).unsqueeze(0)
        c = torch.tanh(self.cell_state_init_layer(features_mean)).unsqueeze(0)
        return c, h

    def project_features(self, features):
        features_flat = features.view(-1, self.D)
        features_proj = F.relu(self.feats_proj_layer(features_flat))
        features_proj = features_proj.view(-1, self.L, self.D)
        return features_proj

    def batch_norm(self, x):
        return self.features_batch_norm(x)

    def _word_embedding(self, inputs):
        embed_inputs = self.embedding_lookup(inputs)  # (N, T, M) or (N, M)
        return embed_inputs

    def _attention_layer(self, features, features_proj, hidden_states):
        h_att = F.relu(features_proj + self.hidden_to_attention_layer(hidden_states.squeeze(0)).unsqueeze(1))    # (N, L, D)
        out_att = self.attention_layer(h_att.view(-1, self.D)).view(-1, self.L)   # (N, L)
        alpha = F.softmax(out_att, dim=-1)
        context = torch.sum(features * alpha.unsqueeze(2), 1)   #(N, D)
        return context, alpha

    def _selector(self, context, hidden_state):
        beta = torch.sigmoid(self.selector_layer(hidden_state.squeeze(0)))    # (N, 1)
        context = context * beta
        return context, beta

    def _decode_lstm(self, x, h, context):
        h = self.dropout(h)
        h_logits = self.hidden_to_embedding_layer(h)

        if self.ctx2out:
            h_logits += self.context_to_embedding_layer(context)

        if self.prev2out:
            h_logits += x
        h_logits = torch.tanh(h_logits)

        h_logits = self.dropout(h_logits)
        out_logits = self.embedding_to_output_layer(h_logits)
        return out_logits
    
    def forward(self, features, features_proj, past_captions, hidden_states, cell_states):
        emb_captions = self._word_embedding(inputs=past_captions)

        context, alpha = self._attention_layer(features, features_proj, hidden_states)

        if self.enable_selector:
            context, beta = self._selector(context, hidden_states)

        next_input = torch.cat((emb_captions, context), 1).unsqueeze(0)

        output, (next_hidden_states, next_cell_states) = self.lstm_cell(next_input, (hidden_states, cell_states))

        logits = self._decode_lstm(emb_captions, output, context)

        return logits, alpha, (next_hidden_states, next_cell_states)
