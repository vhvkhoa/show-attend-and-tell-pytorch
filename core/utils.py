import numpy as np
import json
import time
import os
import sys
import logging
sys.path.append('../coco-caption')
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

def decode_captions(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        print(i)
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == '<END>':
                words.append('.')
                break
            if word != '<NULL>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded


def sample_coco_minibatch(data, batch_size):
    data_size = data['n_examples']
    mask = np.random.choice(data_size, batch_size)
    file_names = data['file_name'][mask]
    return mask, file_names


def write_scores(scores, path, epoch, iteration):
    with open(os.path.join(path, 'val.bleu.scores.txt'), 'a') as f:
        f.write('Epoch %d. Iteration %d\n' %(epoch+1, iteration+1))
        f.write('Bleu_1: %f\n' %scores['Bleu_1'])
        f.write('Bleu_2: %f\n' %scores['Bleu_2'])
        f.write('Bleu_3: %f\n' %scores['Bleu_3'])
        f.write('Bleu_4: %f\n' %scores['Bleu_4'])
        f.write('METEOR: %f\n' %scores['METEOR'])
        f.write('ROUGE_L: %f\n' %scores['ROUGE_L'])
        f.write('CIDEr: %f\n\n' %scores['CIDEr'])

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def evaluate(data_path='./data', split='val', get_scores=False):
    reference_path = os.path.join(data_path, "annotations/captions_%s2017.json" %(split))
    candidate_path = os.path.join(data_path, "%s/%s.candidate.captions.json" %(split, split))

    # load caption data
    ref = COCO(reference_path)
    hypo = ref.loadRes(candidate_path)

    cocoEval = COCOEvalCap(ref, hypo)
    cocoEval.evaluate()
    final_scores = {}
    for metric, score in cocoEval.eval.items():
        final_scores[metric] = score
        logging.info('%s:\t%.3f'%(metric, score))

    if get_scores:
        return final_scores
