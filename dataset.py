from torch.utils.data import Dataset
import pandas as pd
from torchtext.data import get_tokenizer
from torchtext.vocab import vocab
from collections import Counter
import numpy as np
import typing
from tqdm import tqdm
import random
import torch


class IMDBBertDataset(Dataset):
    # Define Special tokens as attributes of class
    CLS = '[CLS]'
    PAD = '[PAD]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    UNK = '[UNK]'

    MASK_PERCENTAGE = 0.15  # How much words to mask

    MASKED_INDICES_COLUMN = 'masked_indices'
    TARGET_COLUMN = 'indices'
    NSP_TARGET_COLUMN = 'is_next'
    TOKEN_MASK_COLUMN = 'token_mask'

    OPTIMAL_LENGTH_PERCENTILE = 70

    def __init__(self, path, ds_from=None, ds_to=None, should_include_text=False):
        self.ds: pd.Series = pd.read_csv(path, engine='python')['review']

        if ds_from is not None or ds_to is not None:
            self.ds = self.ds[ds_from:ds_to]

        self.tokenizer = get_tokenizer('basic_english')
        self.counter = Counter()
        self.vocab = None

        self.optimal_sentence_length = None
        self.should_include_text = should_include_text

        if should_include_text:
            self.columns = ['masked_sentence', self.MASKED_INDICES_COLUMN, 'sentence', self.TARGET_COLUMN,
                            self.TOKEN_MASK_COLUMN,
                            self.NSP_TARGET_COLUMN]
        else:
            self.columns = [self.MASKED_INDICES_COLUMN, self.TARGET_COLUMN, self.TOKEN_MASK_COLUMN,
                            self.NSP_TARGET_COLUMN]
        self.df = self.prepare_dataset()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        inp = torch.Tensor(item[self.MASKED_INDICES_COLUMN]).long()
        token_mask = torch.Tensor(item[self.TOKEN_MASK_COLUMN]).bool()

        attention_mask = (inp == self.vocab[self.PAD]).unsqueeze(0)

        # NSP target
        if item[self.NSP_TARGET_COLUMN] == 0:
            t = [1, 0]
        else:
            t = [0, 1]
        nsp_target = torch.Tensor(t)

        # MLM target
        mask_target = torch.Tensor(item[self.TARGET_COLUMN]).long()
        mask_target = mask_target.masked_fill_(token_mask, 0)
        return inp, attention_mask, token_mask, mask_target, nsp_target

    def prepare_dataset(self) -> pd.DataFrame:
        sentences = []
        nsp = []
        sentence_lens = []

        # Split sentences from dataset:
        for review in self.ds:
            review_sentences = review.split(".")
            sentences += review_sentences
            self._update_length(review_sentences, sentence_lens)
        self.optimal_sentence_length = self._find_optimal_sentence_length(sentence_lens)

        # Create vocabulary:
        print("Create vocabulary")
        for sentence in tqdm(sentences):
            s = self.tokenizer(sentence)
            self.counter.update(s)
        self._fill_vocab()

        # Create training dataset:
        print("Preprocessing dataset")
        for review in tqdm(self.ds):
            review_sentences = review.split('.')
            for i in range(len(review_sentences)-1):
                # True NSP item
                first, second = self.tokenizer(review_sentences[i]), self.tokenizer(review_sentences[i+1])
                nsp.append(self._create_item(first, second, 1))

                # False NSP item
                first, second = self._select_false_nsp_sentences(sentences)
                first, second = self.tokenizer(first), self.tokenizer(second)
                nsp.append(self._create_item(first, second, 0))
        df = pd.DataFrame(nsp, columns=self.columns)
        return df

    def _update_length(self, review_sentences, sentence_lens):
        for word in review_sentences:
            sentence_lens.append(len(word))

    def _find_optimal_sentence_length(self, lengths: typing.List[int]):
        arr = np.array(lengths)
        return int(np.percentile(arr, self.OPTIMAL_LENGTH_PERCENTILE))

    def _fill_vocab(self):
        # specials= argument is only in 0.12.0 version
        # specials=[self.CLS, self.PAD, self.MASK, self.SEP, self.UNK]
        self.vocab = vocab(self.counter, min_freq=2)
        # 0.11.0 uses this approach to insert specials
        self.vocab.insert_token(self.CLS, 0)
        self.vocab.insert_token(self.PAD, 1)
        self.vocab.insert_token(self.MASK, 2)
        self.vocab.insert_token(self.SEP, 3)
        self.vocab.insert_token(self.UNK, 4)
        self.vocab.set_default_index(4)

    def _create_item(self, first: typing.List[str], second: typing.List[str], target: int = 1):
        # Create masked sentence item
        updated_first, first_mask = self._preprocess_sentence(first.copy())
        updated_second, second_mask = self._preprocess_sentence(second.copy())
        nsp_sentence = updated_first + [self.SEP] + updated_second
        nsp_indices = self.vocab.lookup_indices(nsp_sentence)
        inverse_token_mask = first_mask + [True] + second_mask

        # Create sentence item without masking random words
        first, _ = self._preprocess_sentence(first.copy(), should_mask=False)
        second, _ = self._preprocess_sentence(second.copy(), should_mask=False)
        original_nsp_sentence = first + [self.SEP] + second
        original_nsp_indices = self.vocab.lookup_indices(original_nsp_sentence)

        if self.should_include_text:
            return [nsp_sentence, nsp_indices, original_nsp_sentence, original_nsp_indices, inverse_token_mask, target]
        else:
            return [nsp_indices, original_nsp_indices, inverse_token_mask, target]

    def _select_false_nsp_sentences(self, sentences):
        return random.choice(sentences), random.choice(sentences)

    def _preprocess_sentence(self, sentence, should_mask = True):
        inverse_token_mask = [True for _ in range(max(len(sentence), self.optimal_sentence_length))]
        if should_mask:
            sentence, inverse_token_mask = self._mask_sentence(sentence)
        return self._pad_sentence(sentence, inverse_token_mask)

    # Step 1: Mask sentence
    def _mask_sentence(self, sentence: typing.List[str]):
        len_s = len(sentence)
        inverse_token_mask = [True for _ in range(max(len_s, self.optimal_sentence_length))]

        mask_amount = round(len_s * self.MASK_PERCENTAGE)
        for _ in range(mask_amount):
            i = random.randint(0, len_s - 1)
            j = random.randint(5, len(self.vocab)-1)

            if random.random() < 0.8:
                sentence[i] = self.MASK
            else:
                sentence[i] = self.vocab.lookup_token(j)
            inverse_token_mask[i] = False

        return sentence, inverse_token_mask

    # Step 2:Preprocessing: [CLS] and [PAD] sentence
    def _pad_sentence(self, sentence: typing.List[str], inverse_token_mask: typing.List[bool] = None):
        len_s = len(sentence)

        if len_s >= self.optimal_sentence_length:
            s = sentence[:self.optimal_sentence_length]
        else:
            s = sentence + [self.PAD] * (self.optimal_sentence_length - len_s)

        # inverse token mask should be padded as well
        if inverse_token_mask:
            len_m = len(inverse_token_mask)
            if len_m >= self.optimal_sentence_length:
                inverse_token_mask = inverse_token_mask[:self.optimal_sentence_length]
            else:
                inverse_token_mask = inverse_token_mask + [True] * (self.optimal_sentence_length - len_m)
        return s, inverse_token_mask