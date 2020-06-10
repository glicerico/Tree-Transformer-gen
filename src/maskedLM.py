import argparse
import sys

import numpy as np
import torch

from test_gen import SentenceGenerator
sys.path.insert(1, '/home/andres/repositories/Tree-Transformer')  # insert at 1 (0 is the script path (or '' in REPL))
from utils import cc


def parse():
    parser = argparse.ArgumentParser(description="tree transformer")
    parser.add_argument('-no_cuda', action='store_true', help="Don't use GPUs.")
    parser.add_argument('-model_dir', default='train_model', help='output model weight dir')
    parser.add_argument('-seq_length', type=int, default=50, help='sequence length')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_step', type=int, default=100000, help='sequence length')
    parser.add_argument('-data_dir', default='data_dir', help='data dir')
    parser.add_argument('-load', action='store_true', help='load pretrained model')
    parser.add_argument('-train', action='store_true', help='whether train the model')
    parser.add_argument('-test', action='store_true', help='whether test')
    parser.add_argument('-valid_path', default='data/valid.txt', help='validation data path')
    parser.add_argument('-train_path', default='data/train.txt', help='training data path')
    parser.add_argument('-test_path', default='data/test.txt', help='testing data path')
    parser.add_argument('-mode', default='sequential', help='generation mode')
    args = parser.parse_args()

    return args


def print_top_predictions(probs, tokenizer, k=5):
    probs = probs.detach().numpy()
    top_indexes = np.argpartition(probs, -k)[-k:]
    sorted_indexes = top_indexes[np.argsort(-probs[top_indexes])]
    top_tokens = tokenizer.convert_ids_to_tokens(sorted_indexes)
    print(f"Ordered top predicted tokens: {top_tokens}")
    print(f"Ordered top predicted values: {probs[sorted_indexes]}")


if __name__ == '__main__':
    args = parse()
    sent_gen = SentenceGenerator(args)

    sent = "This is a [MASK] ."
    mask_pos = 4
    tok_sent = ['[CLS]']
    tok_sent.extend(sent_gen.data_utils.tokenizer.tokenize(sent))
    tok_sent.append('[SEP]')
    inp = cc([sent_gen.data_utils.tokenizer.encode(sent, add_special_tokens=True)], args.no_cuda)
    mask_sent = np.expand_dims(inp != 102, -2).astype(np.int32)
    print(tok_sent)
    print(inp)
    print(mask_sent)

    predictions, break_probs = sent_gen.model.forward(inp.long(), cc(mask_sent, sent_gen.no_cuda)[0])
    sm = torch.nn.Softmax(dim=0)  # Used to convert logits to probs
    for pos in range(1, inp.shape[1]):
        print(f"Prediction for word: {tok_sent[pos]}")
        probs = sm(predictions[0, pos])
        print_top_predictions(probs, sent_gen.data_utils.tokenizer)

    # for sent in bert_sents:
    #     sent_gen.printer(sent, should_detokenize=True)
