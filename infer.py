import cPickle as pickle
from tensorflow import flags
from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
from core.utils import evaluate

FLAGS = flags.FLAGS

"""Model's parameters"""
flags.DEFINE_integer('image_feature_size', 196, 'Multiplication of width and height of image feature\'s dimension, e.g 14x14=196 in the original paper.')
flags.DEFINE_integer('image_feature_depth', 1024, 'Depth dimension of image feature, e.g 512 if you extract features at conv-5 of VGG-16 model.')
flags.DEFINE_integer('lstm_hidden_size', 1536, 'Hidden layer size for LSTM cell.')
flags.DEFINE_integer('time_steps', 31, 'Number of time steps to be iterating through.')
flags.DEFINE_integer('embed_dim', 512, 'Embedding space size for embedding tokens.')
flags.DEFINE_integer('beam_size', 3, 'Beam size for inference phase.')
flags.DEFINE_float('dropout', 0.5, 'Dropout portion.')
flags.DEFINE_boolean('prev2out', True, 'Link previous hidden state to output.')
flags.DEFINE_boolean('ctx2out', True, 'Link context features to output.')
flags.DEFINE_boolean('enable_selector', True, 'Enable selector to determine how much important the image context is at every time step.')

"""Other parameters"""
flags.DEFINE_boolean('att_vis', False, 'Attention visualization, will show attention masks of every word.') 
flags.DEFINE_string('test_checkpoint', '', 'Path to a checkpoint used to infer.') 
flags.DEFINE_string('word_to_idx_dict', 'word_to_idx.pkl', 'Path to pickle file contained dictionary of words and their corresponding indices.')
flags.DEFINE_string('split', 'val', 'Split contained extracted features of images you want to caption.\n' + 
                                    'Split should be inside ./data/ repository, if not, an error would be raised.\n' +
                                    'Features can be extracted by running prepro.py file.\n'+
                                    'Run python prepro.py to see instructions.')
flags.DEFINE_integer('batch_size', 128, 'Number of examples per mini-batch.')

def main():
    # load dataset and vocab
    data = load_coco_data(data_path='./data', split=FLAGS.split)
    with open('./data/train/word_to_idx.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)

    model = CaptionGenerator(word_to_idx, dim_feature=[FLAGS.image_feature_size, FLAGS.image_feature_depth], dim_embed=FLAGS.embed_dim,
                                    dim_hidden=FLAGS.lstm_hidden_size, n_time_step=FLAGS.time_steps, prev2out=FLAGS.prev2out,
                                    ctx2out=FLAGS.ctx2out, alpha_c=1.0, enable_selector=FLAGS.enable_selector, dropout=FLAGS.dropout)

    solver = CaptioningSolver(model, batch_size=FLAGS.batch_size, 
                                    test_checkpoint=FLAGS.test_checkpoint)

    solver.test(data, beam_size=3, attention_visualization=FLAGS.att_vis)

    #evaluate()

if __name__ == "__main__":
    main()
