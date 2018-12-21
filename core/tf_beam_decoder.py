"""
Beam decoder for tensorflow

Sample usage:

```
from tf_beam_decoder import  beam_decoder

decoded_sparse, decoded_logprobs = beam_decoder(
    cell=cell,
    beam_size=7,
    stop_token=2,
    initial_state=initial_state,
    initial_input=initial_input,
    tokens_to_inputs_fn=lambda tokens: tf.nn.embedding_lookup(my_embedding, tokens),
)
```

See the `beam_decoder` function for complete documentation. (Only the
`beam_decoder` function is part of the public API here.)
"""
import tensorflow as tf
import numpy as np

from tensorflow.python.util import nest

# %%

def nest_map(func, nested):
    if not nest.is_sequence(nested):
        return func(nested)
    flat = nest.flatten(nested)
    return nest.pack_sequence_as(nested, list(map(func, flat)))

# %%

def sparse_boolean_mask(tensor, mask):
    """
    Creates a sparse tensor from masked elements of `tensor`

    Inputs:
      tensor: a 2-D tensor, [batch_size, T]
      mask: a 2-D mask, [batch_size, T]

    Output: a 2-D sparse tensor
    """
    mask_lens = tf.reduce_sum(tf.cast(mask, tf.int32), -1, keep_dims=True)
    mask_shape = tf.shape(mask)
    left_shifted_mask = tf.tile(
        tf.expand_dims(tf.range(mask_shape[1]), 0),
        [mask_shape[0], 1]
    ) < mask_lens
    return tf.SparseTensor(
        indices=tf.where(left_shifted_mask),
        values=tf.boolean_mask(tensor, mask),
        shape=tf.cast(tf.stack([mask_shape[0], tf.reduce_max(mask_lens)]), tf.int64) # For 2D only
    )

# %%

def flat_batch_gather(flat_params, indices, validate_indices=None,
    batch_size=None,
    options_size=None):
    """
    Gather slices from `flat_params` according to `indices`, separately for each
    example in a batch.

    output[(b * indices_size + i), :, ..., :] = flat_params[(b * options_size + indices[b, i]), :, ..., :]

    The arguments `batch_size` and `options_size`, if provided, are used instead
    of looking up the shape from the inputs. This may help avoid redundant
    computation (TODO: figure out if tensorflow's optimizer can do this automatically)

    Args:
      flat_params: A `Tensor`, [batch_size * options_size, ...]
      indices: A `Tensor`, [batch_size, indices_size]
      validate_indices: An optional `bool`. Defaults to `True`
      batch_size: (optional) an integer or scalar tensor representing the batch size
      options_size: (optional) an integer or scalar Tensor representing the number of options to choose from
    """
    if batch_size is None:
        batch_size = indices.get_shape()[0].value
        if batch_size is None:
            batch_size = tf.shape(indices)[0]

    if options_size is None:
        options_size = flat_params.get_shape()[0].value
        if options_size is None:
            options_size = tf.shape(flat_params)[0] // batch_size
        else:
            options_size = options_size // batch_size

    indices_offsets = tf.reshape(tf.range(batch_size) * options_size, [-1] + [1] * (len(indices.get_shape())-1))
    indices_into_flat = indices + tf.cast(indices_offsets, indices.dtype)
    flat_indices_into_flat = tf.reshape(indices_into_flat, [-1])

    return tf.gather(flat_params, flat_indices_into_flat, validate_indices=validate_indices)

def batch_gather(params, indices, validate_indices=None,
    batch_size=None,
    options_size=None):
    """
    Gather slices from `params` according to `indices`, separately for each
    example in a batch.

    output[b, i, ..., j, :, ..., :] = params[b, indices[b, i, ..., j], :, ..., :]

    The arguments `batch_size` and `options_size`, if provided, are used instead
    of looking up the shape from the inputs. This may help avoid redundant
    computation (TODO: figure out if tensorflow's optimizer can do this automatically)

    Args:
      params: A `Tensor`, [batch_size, options_size, ...]
      indices: A `Tensor`, [batch_size, ...]
      validate_indices: An optional `bool`. Defaults to `True`
      batch_size: (optional) an integer or scalar tensor representing the batch size
      options_size: (optional) an integer or scalar Tensor representing the number of options to choose from
    """
    if batch_size is None:
        batch_size = params.get_shape()[0].merge_with(indices.get_shape()[0]).value
        if batch_size is None:
            batch_size = tf.shape(indices)[0]

    if options_size is None:
        options_size = params.get_shape()[1].value
        if options_size is None:
            options_size = tf.shape(params)[1]

    batch_size_times_options_size = batch_size * options_size

    # TODO(nikita): consider using gather_nd. However as of 1/9/2017 gather_nd
    # has no gradients implemented.
    flat_params = tf.reshape(params, tf.concat([[batch_size_times_options_size], tf.shape(params)[2:]], 0))

    indices_offsets = tf.reshape(tf.range(batch_size) * options_size, [-1] + [1] * (len(indices.get_shape())-1))
    indices_into_flat = indices + tf.cast(indices_offsets, indices.dtype)

    return tf.gather(flat_params, indices_into_flat, validate_indices=validate_indices)

# %%

class BeamFlattenWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell, beam_size):
        self.cell = cell
        self.beam_size = beam_size

    def merge_batch_beam(self, tensor):
        remaining_shape = tf.shape(tensor)[2:]
        res = tf.reshape(tensor, tf.concat([[-1], remaining_shape], 0))
        res.set_shape(tf.TensorShape((None,)).concatenate(tensor.get_shape()[2:]))
        return res

    def unmerge_batch_beam(self, tensor):
        remaining_shape = tf.shape(tensor)[1:]
        res = tf.reshape(tensor, tf.concat([[-1, self.beam_size], remaining_shape], 0))
        res.set_shape(tf.TensorShape((None,self.beam_size)).concatenate(tensor.get_shape()[1:]))
        return res

    def prepend_beam_size(self, element):
        return tf.TensorShape(self.beam_size).concatenate(element)

    def tile_along_beam(self, state):
        if nest.is_sequence(state):
            return nest_map(
                lambda val: self.tile_along_beam(val),
                state
            )

        if not isinstance(state, tf.Tensor):
            raise ValueError("State should be a sequence or tensor")

        tensor = state

        tensor_shape = tensor.get_shape().with_rank_at_least(1)
        new_tensor_shape = tensor_shape[:1].concatenate(self.beam_size).concatenate(tensor_shape[1:])

        dynamic_tensor_shape = tf.unstack(tf.shape(tensor))
        res = tf.expand_dims(tensor, 1)
        res = tf.tile(res, [1, self.beam_size] + [1] * (tensor_shape.ndims-1))
        res = tf.reshape(res, [-1, self.beam_size] + list(dynamic_tensor_shape[1:]))
        res.set_shape(new_tensor_shape)
        return res

    def __call__(self, inputs, state, scope=None):
        flat_inputs = nest_map(self.merge_batch_beam, inputs)
        flat_state = nest_map(self.merge_batch_beam, state)

        flat_output, flat_next_state = self.cell(flat_inputs, flat_state, scope=scope)

        output = nest_map(self.unmerge_batch_beam, flat_output)
        next_state = nest_map(self.unmerge_batch_beam, flat_next_state)

        return output, next_state

    @property
    def state_size(self):
        return nest_map(self.prepend_beam_size, self.cell.state_size)

    @property
    def output_size(self):
        return nest_map(self.prepend_beam_size, self.cell.output_size)

# %%

class BeamReplicateWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell, beam_size):
        self.cell = cell
        self.beam_size = beam_size

    def prepend_beam_size(self, element):
        return tf.TensorShape(self.beam_size).concatenate(element)

    def tile_along_beam(self, state):
        if nest.is_sequence(state):
            return nest_map(
                lambda val: self.tile_along_beam(val),
                state
            )

        if not isinstance(state, tf.Tensor):
            raise ValueError("State should be a sequence or tensor")

        tensor = state

        tensor_shape = tensor.get_shape().with_rank_at_least(1)
        new_tensor_shape = tensor_shape[:1].concatenate(self.beam_size).concatenate(tensor_shape[1:])

        dynamic_tensor_shape = tf.unstack(tf.shape(tensor))
        res = tf.expand_dims(tensor, 1)
        res = tf.tile(res, [1, self.beam_size] + [1] * (tensor_shape.ndims-1))
        res = tf.reshape(res, [-1, self.beam_size] + list(dynamic_tensor_shape[1:]))
        res.set_shape(new_tensor_shape)
        return res

    def __call__(self, inputs, state, scope=None):
        varscope = scope or tf.get_variable_scope()

        flat_inputs = nest.flatten(inputs)
        flat_state = nest.flatten(state)

        flat_inputs_unstacked = list(zip(*[tf.unstack(tensor, num=self.beam_size, axis=1) for tensor in flat_inputs]))
        flat_state_unstacked = list(zip(*[tf.unstack(tensor, num=self.beam_size, axis=1) for tensor in flat_state]))

        flat_output_unstacked = []
        flat_next_state_unstacked = []
        output_sample = None
        next_state_sample = None

        for i, (inputs_k, state_k) in enumerate(zip(flat_inputs_unstacked, flat_state_unstacked)):
            inputs_k = nest.pack_sequence_as(inputs, inputs_k)
            state_k = nest.pack_sequence_as(state, state_k)

            # TODO(nikita): is this scope stuff correct?
            if i == 0:
                output_k, next_state_k = self.cell(inputs_k, state_k, scope=scope)
            else:
                with tf.variable_scope(varscope, reuse=True):
                    output_k, next_state_k = self.cell(inputs_k, state_k, scope=varscope if scope is not None else None)

            flat_output_unstacked.append(nest.flatten(output_k))
            flat_next_state_unstacked.append(nest.flatten(next_state_k))

            output_sample = output_k
            next_state_sample = next_state_k


        flat_output = [tf.stack(tensors, axis=1) for tensors in zip(*flat_output_unstacked)]
        flat_next_state = [tf.stack(tensors, axis=1) for tensors in zip(*flat_next_state_unstacked)]

        output = nest.pack_sequence_as(output_sample, flat_output)
        next_state = nest.pack_sequence_as(next_state_sample, flat_next_state)

        return output, next_state

    @property
    def state_size(self):
        return nest_map(self.prepend_beam_size, self.cell.state_size)

    @property
    def output_size(self):
        return nest_map(self.prepend_beam_size, self.cell.output_size)

# %%

class BeamSearchHelper(object):
    # Our beam scores are stored in a fixed-size tensor, but sometimes the
    # tensor size is greater than the number of elements actually on the beam.
    # The invalid elements are assigned a highly negative score.
    # However, top_k errors if any of the inputs have a score of -inf, so we use
    # a large negative constant instead
    INVALID_SCORE = -1e18

    def __init__(self, cell, beam_size, start_token, stop_token,
            score_upper_bound=None,
            max_len=100,
            model=None,
            features=None,
            features_proj=None,
            outputs_to_score_fn=None,
            tokens_to_inputs_fn=None,
            selector=True,
            cell_transform='default',
            scope=None
            ):
 
        self.beam_size = beam_size
        self.start_token = start_token
        self.stop_token = stop_token
        self.max_len = max_len
        self.model=model
        self.feats = features
        self.feats_proj = features_proj
        self.selector = selector
        self.scope = scope

        if score_upper_bound is None and outputs_to_score_fn is None:
            self.score_upper_bound = 0.0
        elif score_upper_bound is None or score_upper_bound > 3e38:
            # Note: 3e38 is just a little smaller than the largest float32
            # Second condition allows for Infinity as a synonym for None
            self.score_upper_bound = None
        else:
            self.score_upper_bound = float(score_upper_bound)

        if self.max_len is None and self.score_upper_bound is None:
            raise ValueError("Beam search needs a stopping criterion. Please provide max_len or score_upper_bound.")

        if cell_transform == 'default':
            if type(cell) in [tf.nn.rnn_cell.LSTMCell,
                              tf.nn.rnn_cell.GRUCell,
                              tf.nn.rnn_cell.BasicLSTMCell,
                              tf.nn.rnn_cell.BasicRNNCell]:
                cell_transform = 'flatten'
            else:
                cell_transform = 'replicate'

        if cell_transform == 'flatten':
            self.cell = BeamFlattenWrapper(cell, self.beam_size)
        elif cell_transform == 'replicate':
            self.cell = BeamReplicateWrapper(cell, self.beam_size)
        else:
            raise ValueError("cell_transform must be one of: 'default', 'flatten', 'replicate'")

        self._cell_transform_used = cell_transform

        if outputs_to_score_fn is not None:
            self.outputs_to_score_fn = outputs_to_score_fn
        if tokens_to_inputs_fn is not None:
            self.tokens_to_inputs_fn = tokens_to_inputs_fn

    def outputs_to_score_fn(self, cell_output):
        return tf.nn.log_softmax(cell_output)

    def tokens_to_inputs_fn(self, symbols):
        return tf.expand_dims(symbols, -1)

    def beam_setup(self, time):
        emit_output = None

        init_state = self.model._get_initial_lstm(features=self.feats)
        init_input = self.model._word_embedding(inputs=tf.fill([tf.shape(self.feats)[0]], self.start_token), reuse=False)

        context, alpha = self.model._attention_layer(self.feats, self.feats_proj, init_state[1], reuse=False)

        if self.selector:
            context, beta = self.model._selector(context, init_state[1], reuse=False)

        init_input = tf.concat([init_input, context], 1)

        init_state = self.cell.tile_along_beam(init_state)
        init_state = tf.nn.rnn_cell.LSTMStateTuple(init_state[0], init_state[1])
        init_input = self.cell.tile_along_beam(init_input)
        context = self.cell.tile_along_beam(context)
        
        self.feats=tf.reshape(tf.tile(tf.expand_dims(self.feats, 1), [1, self.beam_size, 1, 1]),
                            [-1] + self.feats.shape[1:].as_list())
        self.feats_proj=tf.reshape(tf.tile(tf.expand_dims(self.feats_proj, 1), [1, self.beam_size, 1, 1]),
                            [-1] + self.feats_proj.shape[1:].as_list())

        batch_size = tf.Dimension(None)
        if not nest.is_sequence(init_state):
            batch_size = batch_size.merge_with(init_state.get_shape()[0])
        else:
            for tensor in nest.flatten(init_state):
                batch_size = batch_size.merge_with(tensor.get_shape()[0])

        if not nest.is_sequence(init_input):
            batch_size = batch_size.merge_with(init_input.get_shape()[0])
        else:
            for tensor in nest.flatten(init_input):
                batch_size = batch_size.merge_with(tensor.get_shape()[0])

        self.inferred_batch_size = batch_size.value
        if self.inferred_batch_size is not None:
            self.batch_size = self.inferred_batch_size
        else:
            if not nest.is_sequence(init_state):
                self.batch_size = tf.shape(init_state)[0]
            else:
                self.batch_size = tf.shape(list(nest.flatten(init_state))[0])[0]

        self.inferred_batch_size_times_beam_size = None
        if self.inferred_batch_size is not None:
            self.inferred_batch_size_times_beam_size = self.inferred_batch_size * self.beam_size

        self.batch_size_times_beam_size = self.batch_size * self.beam_size

        next_cell_state = init_state
        next_input = init_input

        # Set up the beam search tracking state
        cand_symbols = tf.fill([self.batch_size, 0], tf.constant(self.stop_token, dtype=tf.int32))
        cand_logprobs = tf.ones((self.batch_size,), dtype=tf.float32) * -float('inf')
        cand_finished = tf.zeros((self.batch_size,), dtype=tf.bool)

        cand_alphas = tf.reshape(alpha, [-1, 1, alpha.shape[-1]])
        cand_betas = tf.reshape(beta, [-1, 1])

        beam_alphas = tf.reshape(tf.tile(tf.expand_dims(alpha, 1), 
                                        [1, self.beam_size, 1]), 
                                [-1, 1, alpha.shape[-1]])
        beam_betas = tf.reshape(tf.tile(beta, [1, self.beam_size]),
                                [-1, 1])

        first_in_beam_mask = tf.equal(tf.range(self.batch_size_times_beam_size) % self.beam_size, 0)

        beam_symbols = tf.fill([self.batch_size_times_beam_size, 0], tf.constant(self.stop_token, dtype=tf.int32))
        beam_context = tf.reshape(context, [-1, context.shape[-1]])

        beam_logprobs = tf.where(
            first_in_beam_mask,
            tf.fill([self.batch_size_times_beam_size], 0.0),
            tf.fill([self.batch_size_times_beam_size], self.INVALID_SCORE)
        )

        # Set up correct dimensions for maintaining loop invariants.
        # Note that the last dimension (initialized to zero) is not a loop invariant,
        # so we need to clear it. TODO(nikita): is there a public API for clearing shape
        # inference so that _shape is not necessary?
        cand_symbols._shape = tf.TensorShape((self.inferred_batch_size, None))
        cand_logprobs._shape = tf.TensorShape((self.inferred_batch_size,))
        cand_finished._shape = tf.TensorShape((self.inferred_batch_size,))
        beam_symbols._shape = tf.TensorShape((self.inferred_batch_size_times_beam_size, None))
        beam_logprobs._shape = tf.TensorShape((self.inferred_batch_size_times_beam_size,))
        cand_alphas._shape = tf.TensorShape((self.inferred_batch_size, None, cand_alphas.shape[-1]))
        cand_betas._shape = tf.TensorShape((self.inferred_batch_size, None))
        beam_alphas._shape = tf.TensorShape((self.inferred_batch_size_times_beam_size, None, beam_alphas.shape[-1]))
        beam_betas._shape = tf.TensorShape((self.inferred_batch_size_times_beam_size, None))
        next_loop_state = (
            cand_symbols,
            cand_logprobs,
            cand_finished,
            beam_symbols,
            beam_logprobs,
            cand_alphas,
            cand_betas,
            beam_alphas,
            beam_betas,
            beam_context,
        )

        emit_output = tf.zeros(self.cell.output_size)
        elements_finished = tf.zeros([self.batch_size], dtype=tf.bool)

        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    def beam_loop(self, time, cell_output, cell_state, loop_state):
        (
            past_cand_symbols, # [batch_size, time-1]
            past_cand_logprobs,# [batch_size]
            past_cand_finished,# [batch_size]
            past_beam_symbols, # [batch_size*beam_size, time-1], right-aligned
            past_beam_logprobs,# [batch_size*beam_size]
            past_cand_alphas,  # [batch_size, time, ...]
            past_cand_betas,   # [batch_size, time]
            past_beam_alphas,  # [batch_size, ]
            past_beam_betas,
            past_beam_context,
                ) = loop_state

        # We don't actually use this, but emit_output is required to match the
        # cell output size specfication. Otherwise we would leave this as None.
        emit_output = cell_output

        # 1. Get scores for all candidate sequences
        past_symbols = tf.cond(tf.equal(time, 1), 
                                lambda: tf.fill(dims=[self.batch_size_times_beam_size], value=self.start_token),
                                lambda: past_beam_symbols[:, -1])
        logprobs = self.outputs_to_score_fn(self.model, past_symbols, cell_output, past_beam_context, self.beam_size)

        try:
            num_classes = int(logprobs.get_shape()[-1])
        except:
            # Shape inference failed
            num_classes = tf.shape(logprobs)[-1]

        logprobs_batched = tf.reshape(logprobs + tf.expand_dims(tf.reshape(past_beam_logprobs, [self.batch_size, self.beam_size]), 2),
                                      [self.batch_size, self.beam_size * num_classes])

        # 2. Determine which states to pass to next iteration

        # TODO(nikita): consider using slice+fill+concat instead of adding a mask
        nondone_mask = tf.reshape(
            tf.cast(tf.equal(tf.range(num_classes), self.stop_token), tf.float32) * self.INVALID_SCORE,
            [1, 1, num_classes])

        nondone_mask = tf.reshape(tf.tile(nondone_mask, [1, self.beam_size, 1]),
            [-1, self.beam_size*num_classes])


        beam_logprobs, indices = tf.nn.top_k(logprobs_batched + nondone_mask, self.beam_size)
        beam_logprobs = tf.reshape(beam_logprobs, [-1])

        # For continuing to the next symbols
        symbols = indices % num_classes # [batch_size, self.beam_size]
        parent_refs = indices // num_classes # [batch_size, self.beam_size]

        symbols_history = flat_batch_gather(past_beam_symbols, parent_refs, batch_size=self.batch_size, options_size=self.beam_size)
        beam_symbols = tf.concat([symbols_history, tf.reshape(symbols, [-1, 1])], 1)

        cell_output = tf.reshape(cell_output, [-1, cell_output.shape[-1]])
        cell_output = flat_batch_gather(cell_output, parent_refs, batch_size=self.batch_size, options_size=self.beam_size)

        # Handle the output and the cell state shuffling
        next_cell_state = nest_map(
            lambda element: batch_gather(element, parent_refs, batch_size=self.batch_size, options_size=self.beam_size),
            cell_state
        )

        next_input, context, alpha, beta = self.tokens_to_inputs_fn(self.model, symbols, self.feats, self.feats_proj, cell_output,
                                                                    self.beam_size)

        beam_context = context
        alphas_history = flat_batch_gather(past_beam_alphas, parent_refs, batch_size=self.batch_size, options_size=self.beam_size)
        betas_history = flat_batch_gather(past_beam_betas, parent_refs, batch_size=self.batch_size, options_size=self.beam_size)
        beam_alphas = tf.concat([alphas_history, tf.expand_dims(alpha, 1)], 1)
        beam_betas = tf.concat([betas_history, beta], 1)

        # 3. Update the candidate pool to include entries that just ended with a stop token
        logprobs_done = tf.reshape(logprobs_batched, [-1, self.beam_size, num_classes])[:,:,self.stop_token]
        done_parent_refs = tf.argmax(logprobs_done, 1)

        done_symbols = flat_batch_gather(past_beam_symbols, done_parent_refs, batch_size=self.batch_size, options_size=self.beam_size)
        done_alphas =  flat_batch_gather(past_beam_alphas, done_parent_refs, batch_size=self.batch_size, options_size=self.beam_size)
        done_betas =  flat_batch_gather(past_beam_betas, done_parent_refs, batch_size=self.batch_size, options_size=self.beam_size)

        logprobs_done_max = tf.reduce_max(logprobs_done, 1)

        cand_finished = tf.greater(tf.reduce_max(logprobs_done, 1), tf.reshape(beam_logprobs, [-1, self.beam_size])[:, -1])
        cand_mask = tf.logical_and(tf.logical_not(tf.logical_xor(cand_finished, past_cand_finished)),
                                    logprobs_done_max > past_cand_logprobs)
        cand_mask = tf.logical_and(tf.logical_and(tf.logical_not(cand_finished), past_cand_finished), cand_mask)
        cand_mask =  tf.logical_or(tf.logical_and(tf.logical_not(past_cand_finished), cand_finished), cand_mask)

        cand_finished = tf.logical_or(cand_finished, past_cand_finished)

        cand_symbols_unpadded = tf.where(cand_mask,
                                done_symbols,
                                past_cand_symbols)
        cand_alphas_unpadded = tf.where(cand_mask,
                                done_alphas,
                                past_cand_alphas)
        cand_betas_unpadded = tf.where(cand_mask,
                                done_betas,
                                past_cand_betas)

        cand_logprobs = tf.maximum(logprobs_done_max, past_cand_logprobs)

        cand_symbols = tf.concat([cand_symbols_unpadded, tf.fill([self.batch_size, 1], self.stop_token)], 1)
        cand_alphas = tf.concat([cand_alphas_unpadded, tf.fill([self.batch_size, 1, past_cand_alphas.shape[-1]], 0.)], 1)
        cand_betas = tf.concat([cand_betas_unpadded, tf.fill([self.batch_size, 1], 0.)], 1)
        # 4. Check the stopping criteria

        if self.max_len is not None:
            elements_finished_clip = (time >= self.max_len)

        if self.score_upper_bound is not None:
            elements_finished_bound = tf.reduce_max(tf.reshape(beam_logprobs, [-1, self.beam_size]), 1) < (cand_logprobs - self.score_upper_bound)

        if self.max_len is not None and self.score_upper_bound is not None:
            elements_finished = elements_finished_clip | elements_finished_bound
        elif self.score_upper_bound is not None:
            elements_finished = elements_finished_bound
        elif self.max_len is not None:
            # this broadcasts elements_finished_clip to the correct shape
            elements_finished = tf.zeros([self.batch_size], dtype=tf.bool) | elements_finished_clip
        else:
            assert False, "Lack of stopping criterion should have been caught in constructor"

        # 5. Prepare return values
        # While loops require strict shape invariants, so we manually set shapes
        # in case the automatic shape inference can't calculate these. Even when
        # this is redundant is has the benefit of helping catch shape bugs.

        for tensor in list(nest.flatten(next_input)) + list(nest.flatten(next_cell_state)):
            tensor.set_shape(tf.TensorShape((self.inferred_batch_size, self.beam_size)).concatenate(tensor.get_shape()[2:]))

        for tensor in [cand_symbols, cand_alphas, cand_betas, cand_logprobs, elements_finished]:
            tensor.set_shape(tf.TensorShape((self.inferred_batch_size,)).concatenate(tensor.get_shape()[1:]))

        for tensor in [beam_symbols, beam_alphas, beam_betas, beam_logprobs]:
            tensor.set_shape(tf.TensorShape((self.inferred_batch_size_times_beam_size,)).concatenate(tensor.get_shape()[1:]))

        next_loop_state = (
            cand_symbols,
            cand_logprobs,
            cand_finished,
            beam_symbols,
            beam_logprobs,
            cand_alphas,
            cand_betas,
            beam_alphas,
            beam_betas,
            beam_context,
        )

        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    def loop_fn(self, time, cell_output, cell_state, loop_state):
        if cell_output is None:
            return self.beam_setup(time)
        else:
            return self.beam_loop(time, cell_output, cell_state, loop_state)

    def decode_dense(self):
        emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(self.cell, self.loop_fn, scope=self.scope)
        cand_symbols, cand_logprobs, cand_finished, beam_symbols, beam_logprobs, cand_alphas, cand_betas, beam_alphas, beam_betas, beam_context = final_loop_state
        return cand_symbols, cand_logprobs, cand_alphas, cand_betas 

    def decode_sparse(self, include_stop_tokens=True):
        dense_symbols, logprobs, alphas, betas = self.decode_dense()
        mask = tf.not_equal(dense_symbols, self.stop_token)
        if include_stop_tokens:
            mask = tf.concat([tf.ones_like(mask[:,:1]), mask[:,:-1]], 1)
        return sparse_boolean_mask(dense_symbols, mask), logprobs
# %%

def beam_decoder(
        cell,
        beam_size,
        start_token,
        stop_token,
        tokens_to_inputs_fn,
        outputs_to_score_fn=None,
        score_upper_bound=None,
        max_len=None,
        model=None,
        features=None,
        features_proj=None,
        selector=True,
        cell_transform='default',
        output_dense=False,
        scope=None
        ):
    """Beam search decoder

    Args:
        cell: tf.nn.rnn_cell.RNNCell defining the cell to use
        beam_size: the beam size for this search
        stop_token: the index of the symbol used to indicate the end of the
            output
        tokens_to_inputs_fn: function to go from token numbers to cell inputs.
            A typical implementation would look up the tokens in an embedding
            matrix.
            (signature: [batch_size, beam_size, num_classes] int32 -> [batch_size, beam_size, ...])
        outputs_to_score_fn: function to go from RNN cell outputs to scores for
            different tokens. If left unset, log-softmax is used (i.e. the cell
            outputs are treated as unnormalized logits).
            Inputs to the function are cell outputs, i.e. a possibly nested
            structure of items with shape [batch_size, beam_size, ...].
            Must return a single Tensor with shape [batch_size, beam_size, num_classes]
        score_upper_bound: (float or None). An upper bound on sequence scores.
            Used to determine a stopping criterion for beam search: the search
            stops if the highest-scoring complete sequence found so far beats
            anything on the beam by at least score_upper_bound. For typical
            sequence decoder models, outputs_to_score_fn returns normalized
            logits and this upper bound should be set to 0. Defaults to 0 if
            outputs_to_score_fn is not provided, otherwise defaults to None.
        max_len: (default None) maximum length after which to abort beam search.
            This provides an alternative stopping criterion.
        cell_transform: 'flatten', 'replicate', 'none', or 'default'. Most RNN
            primitives require inputs/outputs/states to have a shape that starts
            with [batch_size]. Beam search instead relies on shapes that start
            with [batch_size, beam_size]. This parameter controls how the arguments
            cell/initial_state/initial_input are transformed to comply with this.
            * 'flatten' creates a virtual batch of size batch_size*beam_size, and
              uses the cell with such a batch size. This transformation is only
              valid for cells that do not rely on the batch ordering in any way.
              (This is true of most RNNCells, but notably excludes cells that
              use attention.)
              The values of initial_state and initial_input are expanded and
              tiled along the beam_size dimension.
            * 'replicate' creates beam_size virtual replicas of the cell, each
              one of which is applied to batch_size elements. This should yield
              correct results (even for models with attention), but may not have
              ideal performance.
              The values of initial_state and initial_input are expanded and
              tiled along the beam_size dimension.
            * 'none' passes along cell/initial_state/initial_input as-is.
              Note that this requires initial_state and initial_input to already
              have a shape [batch_size, beam_size, ...] and a custom cell type
              that can handle this
            * 'default' selects 'flatten' for LSTMCell, GRUCell, BasicLSTMCell,
              and BasicRNNCell. For all other cell types, it selects 'replicate'
        output_dense: (default False) toggles returning the decoded sequence as
            dense tensor.
        scope: VariableScope for the created subgraph; defaults to "RNN".

    Returns:
        A tuple of the form (decoded, log_probabilities) where:
        decoded: A SparseTensor (or dense Tensor if output_dense=True), of
            underlying shape [batch_size, ?] that contains the decoded sequence
            for each batch element
        log_probability: a [batch_size] tensor containing sequence
            log-probabilities
    """
    with tf.variable_scope(scope or "RNN") as varscope:
        helper = BeamSearchHelper(
            cell=cell,
            beam_size=beam_size,
            start_token=start_token,
            stop_token=stop_token,
            tokens_to_inputs_fn=tokens_to_inputs_fn,
            outputs_to_score_fn=outputs_to_score_fn,
            score_upper_bound=score_upper_bound,
            max_len=max_len,
            model=model,
            features=features,
            features_proj=features_proj,
            selector=selector,
            cell_transform=cell_transform,
            scope=varscope
            )

        if output_dense:
            return helper.decode_dense()
        else:
            return helper.decode_sparse()
