# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch
import json

from fairseq.data import data_utils, Dictionary

from . import BaseWrapperDataset, LRUCacheDataset


class MaskSubsTokensDataset(BaseWrapperDataset):
    """
    A wrapper Dataset for masked language modeling.

    Input items are masked according to the specified masking probability.

    Args:
        dataset: Dataset to wrap.
        sizes: Sentence lengths
        vocab: Dictionary with the vocabulary and special tokens.
        pad_idx: Id of pad token in vocab
        mask_idx: Id of mask token in vocab
        return_masked_tokens: controls whether to return the non-masked tokens
            (the default) or to return a tensor with the original masked token
            IDs (and *pad_idx* elsewhere). The latter is useful as targets for
            masked LM training.
        seed: Seed for random number generator for reproducibility.
        subs_prob: probability of replacing a token with *mask_idx*.
        random_token_prob: probability of replacing a masked token with a
            random token from the vocabulary.
        freq_weighted_replacement: sample random replacement words based on
            word frequencies in the vocab.
        mask_whole_words: only mask whole words. This should be a byte mask
            over vocab indices, indicating whether it is the beginning of a
            word. We will extend any mask to encompass the whole word.
        bpe: BPE to use for whole-word masking.
    """

    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, seed: int, *args, **kwargs):
        """Return the source and target datasets for masked LM training."""
        dataset = LRUCacheDataset(dataset)
        return (
            LRUCacheDataset(cls(dataset, seed=seed, *args, **kwargs, return_masked_tokens=False)),
            LRUCacheDataset(cls(dataset, seed=seed, *args, **kwargs, return_masked_tokens=True)),
        )
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        subs: dict,
        pad_idx: int,
        mask_idx: int,
        return_masked_tokens: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        subs_prob: float = 0.3,
        random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
        mask_whole_words: torch.Tensor = None,
    ):
        assert 0.0 < subs_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.subs_prob = subs_prob
        self.random_token_prob = random_token_prob
        self.mask_whole_words = mask_whole_words
        self.subs = subs

        if random_token_prob > 0.0:
            if freq_weighted_replacement:
                weights = np.array(self.vocab.count)
            else:
                weights = np.ones(len(self.vocab))
            weights[:self.vocab.nspecial] = 0
            self.weights = weights / weights.sum()

        self.epoch = 0

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            # mask based on the augmented objective
            item = self.dataset[index]
            sz = len(item)

            assert self.mask_idx not in item, \
                'Dataset contains mask_idx (={}), this is not expected!'.format(
                    self.mask_idx,
                )

            if self.mask_whole_words is not None:
                word_begins_mask = self.mask_whole_words.gather(0, item)
                word_begins_idx = word_begins_mask.nonzero().view(-1)
                sz = len(word_begins_idx)
                words = np.split(word_begins_mask, word_begins_idx)[1:]
                assert len(words) == sz
                word_lens = list(map(len, words))

            # decide elements to mask
            mask = np.full(sz, False)
            num_mask = int(
                # add a random number for probabilistic rounding
                self.mask_prob * sz + np.random.rand()
            )
            mask[np.random.choice(sz, num_mask, replace=False)] = True

            # decide elements to substitute
            tmp_item = []
            for i in item:
                tmp_item.append(str(int(i)))
            switch = []
            tmp = 0
            for i in range(sz):
                if i >= tmp: 
                    # cannot contain the mask
                    for max_len in range(1, min(4, sz - i)):
                        if mask[i + max_len - 1]:
                            max_len -= 1
                            break

                    for len_token in range(max_len, 0, -1):
                        potential_token = " ".join(tmp_item[i:i+len_token])
                        if potential_token in self.subs:
                            switch.append((i, i + len_token - 1, self.subs[potential_token]))
                            tmp = i + len_token
                            break
            #random choose the elements to replace
            num_subs = int(
                # add a random number for probabilistic rounding
                self.subs_prob * sz + np.random.rand()
            )
            if len(switch) < num_subs:
                valid_idxs = [i for i in range(len(switch))]
            else:
                valid_idxs = np.random.choice(len(switch), num_subs, replace=False)
                valid_idxs = sorted(valid_idxs)
            
            # because of the substitution, the position of some masks shifts
            updated_mask = np.array([])
            # replace the elements
            rep_item = torch.LongTensor([])
            pre = 0
            count = 0
            diff = 512 - sz
            for valid_idx in valid_idxs:
                tmp = switch[valid_idx]
                #random pick one substitution from the candidates 
                rand_pick = np.random.choice(len(tmp[2]), 1)[0]
                if len(tmp[2][rand_pick]) - (1 + tmp[1] - tmp[0]) + count <= diff:
                    rep_item = torch.cat((rep_item, item[pre: tmp[0]], torch.LongTensor(tmp[2][rand_pick])), 0)
                    updated_mask = np.append(updated_mask, mask[pre:tmp[0]], np.full(len(tmp[2][rand_pick]), False))
                    count += len(tmp[2][rand_pick]) - (1 + tmp[1] - tmp[0])
                else:
                    rep_item = torch.cat((rep_item, item[pre:tmp[1] + 1]), 0)
                    updated_mask = np.append(updated_mask, mask[pre:tmp[1] + 1])
                pre = tmp[1] + 1
            rep_item = torch.cat((rep_item, item[pre:]), 0)
            updated_mask = np.append(updated_mask, mask[pre:])
            assert len(updated_mask) == len(rep_item)

            item, mask = rep_item, updated_mask
            sz = len(item)
            if self.mask_whole_words is not None:
                word_begins_mask = self.mask_whole_words.gather(0, item)
                word_begins_idx = word_begins_mask.nonzero().view(-1)
                sz = len(word_begins_idx)
                words = np.split(word_begins_mask, word_begins_idx)[1:]
                assert len(words) == sz
                word_lens = list(map(len, words))
 
            if self.return_masked_tokens:
                # exit early if we're just returning the masked tokens
                # (i.e., the targets for masked LM training)
                if self.mask_whole_words is not None:
                    mask = np.repeat(mask, word_lens)
                new_item = np.full(len(mask), self.pad_idx)
                new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
                return torch.from_numpy(new_item)

            # decide unmasking and random replacement
            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            if self.mask_whole_words is not None:
                mask = np.repeat(mask, word_lens)

            new_item = np.copy(item)
            new_item[mask] = self.mask_idx
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    if self.mask_whole_words is not None:
                        rand_mask = np.repeat(rand_mask, word_lens)
                        num_rand = rand_mask.sum()

                    new_item[rand_mask] = np.random.choice(
                        len(self.vocab),
                        num_rand,
                        p=self.weights,
                    )

            return torch.from_numpy(new_item)
