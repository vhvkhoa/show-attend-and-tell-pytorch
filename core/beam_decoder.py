import torch
from torch.nn import functional as F
import numpy as np

class BeamSearchDecoder(object):
    def __init__(self, model, device, beam_size, vocab_size, start_token, stop_token, n_time_steps):
        self.model = model
        self.device = device
        self.beam_size = beam_size
        self.vocab_size = vocab_size
        self._start = start_token
        self._end = stop_token
        self.n_time_steps = n_time_steps
    
    def compute_score(self, logits, beam_logprobs):
        return F.log_softmax(torch.squeeze(logits), dim=-1) + beam_logprobs
    
    def decode(self, features):
        features = features.to(device=self.device)
        features = self.model.batch_norm(features)
        features_proj = self.model.project_features(features)
        hidden_states, cell_states = self.model.get_initial_lstm(features)
        hidden_states.unsqueeze_(0)
        cell_states.unsqueeze_(0)

        batch_size, hidden_layers, hidden_size = features.size(0), hidden_states.size(0), hidden_states.size(-1)

        cand_scores = torch.zeros(batch_size, device=self.device)
        cand_symbols = torch.full([batch_size, self.n_time_steps + 1], self._start, dtype=torch.int64, device=self.device)
        cand_finished = torch.zeros(batch_size, dtype=torch.uint8, device=self.device)

        beam_symbols = torch.full([batch_size, 1, 1], self._start, dtype=torch.int64, device=self.device)
        beam_inputs = torch.full([batch_size, 1], self._start, dtype=torch.int64, device=self.device)
        beam_scores = torch.zeros(batch_size, self.vocab_size, device=self.device)

        for t in range(self.n_time_steps):
            beam_size = beam_inputs.size(1)
            beam_logits, beam_hidden_states, beam_cell_states = [], [], [] 
            for b in range(beam_size):
                logits, alpha, (hidden_states, cell_states) = self.model(features,
                                                                        features_proj,
                                                                        beam_inputs[:, b],
                                                                        hidden_states[b],
                                                                        cell_states[b])
                beam_logits.append(logits)
                beam_hidden_states.append(hidden_states)
                beam_cell_states.append(cell_states)

            beam_logits = torch.flatten(torch.stack(beam_logits, 1), end_dim=1)
            beam_hidden_states = torch.stack(beam_hidden_states)
            beam_cell_states = torch.stack(beam_cell_states)

        
            beam_scores = self.compute_score(beam_logits, beam_scores)
            end_scores = beam_scores[:, self._end].view(batch_size, -1)
            beam_scores = torch.cat([beam_scores[:, :self._end],
                                       beam_scores[:, self._end:]], 1).view(batch_size, -1)

            k_scores, k_indices = torch.topk(beam_scores, self.beam_size)

            # Compute immediate candidate
            done_scores_max, done_parent_indices = torch.max(end_scores, -1)
            done_symbols = torch.cat([torch.gather(beam_symbols, 1,
                                      done_parent_indices.view(-1, 1, 1).repeat(1, 1, t + 1)).squeeze(1),
                                      torch.full([batch_size, self.n_time_steps - t], 
                                            self._end, dtype=torch.int64, device=self.device)], -1)

            cand_mask = (done_scores_max >= k_scores[:, -1]) & (~cand_finished | (done_scores_max > cand_scores))
            cand_finished = cand_mask | cand_finished
            cand_mask = cand_mask.unsqueeze(-1)
            cand_symbols = torch.where(cand_mask, done_symbols, cand_symbols)
            cand_scores = torch.where(cand_mask, done_scores_max, cand_scores)

            # Compute beam candidate for next time-step
            k_symbol_indices = k_indices % self.vocab_size
            k_parent_indices = k_indices // self.vocab_size

            past_beam_symbols = torch.gather(beam_symbols, 1,
                                             k_parent_indices.unsqueeze(-1).repeat(1, 1, t + 1))
            beam_symbols = torch.cat([past_beam_symbols, k_symbol_indices.unsqueeze(-1)], -1)

            k_parent_indices = k_parent_indices.t().unsqueeze(1).unsqueeze(-1).repeat(1, hidden_layers, 1, hidden_size)
            hidden_states = torch.gather(beam_hidden_states, 0, k_parent_indices)
            cell_states = torch.gather(beam_cell_states, 0, k_parent_indices)
            beam_inputs = k_symbol_indices

        # if not finished, get the best sequence in beam candidate
        best_beam_symbols = beam_symbols[:, 0, :]
        cand_symbols = torch.where(cand_finished, cand_symbols, best_beam_symbols)

        return cand_symbols