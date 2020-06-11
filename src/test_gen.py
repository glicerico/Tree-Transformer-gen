import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(1, '/home/andres/repositories/Tree-Transformer')  # insert at 1 (0 is the script path (or '' in REPL))
from solver import Solver
from utils import cc

CLS = '[CLS]'
SEP = '[PAD]'
MASK = '[MASK]'


class SentenceGenerator(Solver):
    def __init__(self, args):
        super(SentenceGenerator, self).__init__(args)

        self.mask_id = self.data_utils.new_vocab[MASK]
        self.sep_id = self.data_utils.new_vocab[SEP]
        self.cls_id = self.data_utils.new_vocab[CLS]

        path = os.path.join(self.model_dir, 'model.pth')
        device = torch.device("cpu" if self.no_cuda else "cuda:0")
        self.model.load_state_dict(torch.load(path, map_location=device)['state_dict'])
        print(f"Loaded model from {path}!")
        self.model.eval()

    def tokenize_batch(self, batch, max_len):
        return [self.data_utils.text2id(" ".join(sent), seq_length=max_len + 2) for sent in batch]

    def untokenize_batch(self, batch):
        return [self.data_utils.id2sent(sent) for sent in batch]

    @staticmethod
    def detokenize(sent):
        """ Roughly detokenizes (mainly undoes wordpiece) """
        new_sent = []
        for i, tok in enumerate(sent):
            if tok.startswith("##"):
                new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
            else:
                new_sent.append(tok)
        return new_sent

    @staticmethod
    def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
        """ Generate a word from from out[gen_idx]

        args:
            - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
            - gen_idx (int): location for which to generate for
            - top_k (int): if >0, only sample from the top k most probable words
            - sample (Bool): if True, sample from full distribution. Overridden by top_k
        """
        logits = out[:, gen_idx]
        if temperature is not None:
            logits = logits / temperature
        if top_k > 0:
            kth_vals, kth_idx = logits.topk(top_k, dim=-1)
            dist = torch.distributions.categorical.Categorical(logits=kth_vals)
            idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
        elif sample:
            dist = torch.distributions.categorical.Categorical(logits=logits)
            idx = dist.sample().squeeze(-1)
        else:
            idx = torch.argmax(logits, dim=-1)
        return idx.tolist() if return_list else idx

    def get_init_text(self, seed_text, max_len, batch_size=1, rand_init=False):
        """ Get initial sentence by padding seed_text with either masks or random words to max_len """
        batch = [seed_text + [MASK] * (max_len - len(seed_text)) + [SEP] for _ in range(batch_size)]
        # if rand_init:
        #    for ii in range(max_len):
        #        init_idx[seed_len+ii] = np.random.randint(0, len(tokenizer.vocab))

        #return self.tokenize_batch(batch)
        return self.tokenize_batch(batch, max_len=max_len)

    def printer(self, sent, should_detokenize=True):
        if should_detokenize:
            sent = self.detokenize(sent)[1:-1]
        print(" ".join(sent))

    """
    This is the meat of the algorithm. The general idea is
        1. start from all masks
        2. repeatedly pick a location, mask the token at that location, and generate from the probability distribution given by BERT
        3. stop when converged or tired of waiting

    We consider three "modes" of generating:
        - generate a single token for a position chosen uniformly at random for a chosen number of time steps
        - generate in sequential order (L->R), one token at a time
        - generate for all positions at once for a chosen number of time steps

    The `generate` function wraps and batches these three generation modes. In practice, we find that the first leads to the most fluent samples.
    """

    def parallel_sequential_generation(self, seed_text, batch_size=10, max_len=15, top_k=0, temperature=None,
                                       max_iter=300,
                                       burnin=200,
                                       print_every=10, verbose=True):
        """ Generate for one random position at a timestep

        args:
            - burnin: during burn-in period, sample from full distribution; afterwards take argmax
        """
        seed_len = len(seed_text)
        batch = self.get_init_text(seed_text, max_len, batch_size)
        inp_mask = []

        for ii in range(max_iter):
            kk = np.random.randint(0, max_len)
            for jj in range(batch_size):
                batch[jj][seed_len + kk] = self.mask_id
            inp = cc(batch, self.no_cuda)
            inp_mask.append(np.expand_dims(inp != self.sep_id, -2).astype(np.int32))
            out, break_probs = self.model(inp, cc(inp_mask, self.no_cuda)[0])
            topk = top_k if (ii >= burnin) else 0
            idxs = self.generate_step(out, gen_idx=seed_len + kk, top_k=topk, temperature=temperature,
                                      sample=(ii < burnin))
            for jj in range(batch_size):
                batch[jj][seed_len + kk] = idxs[jj]

            if verbose and np.mod(ii + 1, print_every) == 0:
                for_print = self.data_utils.id2sent(batch[0]).split()
                for_print = for_print[:seed_len + kk + 1] + ['(*)'] + for_print[seed_len + kk + 1:]
                print("iter", ii + 1, " ".join(for_print))

        return self.untokenize_batch(batch)

    def parallel_generation(self, seed_text, batch_size=10, max_len=15, top_k=0, temperature=None, max_iter=300,
                            sample=True,
                            print_every=10, verbose=True):
        """ Generate for all positions at each time step """
        seed_len = len(seed_text)
        batch = self.get_init_text(seed_text, max_len, batch_size)
        inp_mask = []

        for ii in range(max_iter):
            inp = cc(batch, self.no_cuda)
            inp_mask.append(np.expand_dims(inp != self.sep_id, -2).astype(np.int32))
            out, break_probs = self.model(inp, cc(inp_mask, self.no_cuda)[0])
            for kk in range(max_len):
                idxs = self.generate_step(out, gen_idx=seed_len + kk, top_k=top_k, temperature=temperature,
                                          sample=sample)
                for jj in range(batch_size):
                    batch[jj][seed_len + kk] = idxs[jj]

            if verbose and np.mod(ii, print_every) == 0:
                print("iter", ii + 1, self.data_utils.id2sent(batch[0]))

        return self.untokenize_batch(batch)

    def sequential_generation(self, seed_text, batch_size=10, max_len=15, leed_out_len=15,
                              top_k=0, temperature=None, sample=True):
        """ Generate one word at a time, in L->R order """
        seed_len = len(seed_text)
        batch = self.get_init_text(seed_text, max_len, batch_size)

        for ii in range(max_len - seed_len):
            # inp = [sent[:seed_len + ii + leed_out_len] + [self.sep_id] for sent in batch]
            inp = cc(batch, self.no_cuda)
            inp_mask = [np.expand_dims(i != self.sep_id, -2).astype(np.int32) for i in inp]
            test = cc(inp_mask, self.no_cuda)
            out, break_probs = self.model(inp.long(), cc(inp_mask, self.no_cuda)[0])
            idxs = self.generate_step(out, gen_idx=seed_len + ii, top_k=top_k, temperature=temperature, sample=sample)
            for jj in range(batch_size):
                batch[jj][seed_len + ii] = idxs[jj]

        # return self.untokenize_batch(batch)
        return self.untokenize_batch(batch)

    def generate(self, n_samples, seed_text=CLS, batch_size=10, max_len=25,
                 generation_mode="parallel-sequential",
                 sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500,
                 leed_out_len=15, print_every=1):
        # main generation function to call
        sentences = []
        n_batches = math.ceil(n_samples / batch_size)
        start_time = time.time()
        for batch_n in range(n_batches):
            if generation_mode == "parallel-sequential":
                batch = self.parallel_sequential_generation(seed_text, batch_size=batch_size, max_len=max_len,
                                                            top_k=top_k,
                                                            temperature=temperature, burnin=burnin, max_iter=max_iter,
                                                            verbose=False)
            elif generation_mode == "sequential":
                batch = self.sequential_generation(seed_text, batch_size=batch_size, max_len=max_len, top_k=top_k,
                                                   temperature=temperature, leed_out_len=leed_out_len, sample=sample)
            elif generation_mode == "parallel":
                batch = self.parallel_generation(seed_text, batch_size=batch_size,
                                                 max_len=max_len, top_k=top_k, temperature=temperature,
                                                 sample=sample, max_iter=max_iter,
                                                 verbose=False)

            if (batch_n + 1) % print_every == 0:
                print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
                start_time = time.time()

            sentences += batch
        return sentences

