import argparse
import pathlib
from collections import defaultdict
from itertools import count
import json
import random
import tempfile
import typing

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

import sentencepiece as spm

BOS = "<seq>"
EOS = "</seq>"
PAD = "<pad/>"
UNK = "<unk/>"

SUPPORTED_ARCHS = ("sgns", "char")


class ComprehensiveNGramTokenizer():
    def __init__(self, charset, vocab, n: int = 3):
        self.chardict = {token: idx for idx, token in enumerate(charset)}
        self.vocab = vocab
        self.n = n

        for i in charset:
            for j in charset:
                for k in charset:
                    token = i + j + k

                    if not token in self.vocab:
                        self.vocab[token] = len(self.vocab)

        self.reverse_vocab = {val: key for key, val in self.vocab.items()}

    def encode(self, s: str) -> typing.List:
        if len(s) < self.n:
            s = str(list(s) + [PAD] * (self.n - len(s)))

        result = []
        for i in range(len(s) - self.n + 1):
            substring = s[i:i + self.n]
            result.append(self.vocab[substring])

        return result

    def decode(self, l: typing.List) -> str:
        s = ""

        for idx in range(len(l)):
            substring = self.reverse_vocab[l[idx]]
            s += substring[0]

        s = s[:-1] + self.reverse_vocab[l[len(l) - 1]]

        return s

# Custom Tokenizer to encode ngrams
class AsciiNGramTokenizer():
    def __init__(self, charset, vocab, n: int = 3):
        self.chardict = {token: idx for idx, token in enumerate(charset)}
        self.vocab = vocab
        self.n = n

        for i in charset:
            i_token = i if i.isascii() else UNK
            for j in charset:
                j_token = j if j.isascii() else UNK
                for k in charset:
                    k_token = k if k.isascii() else UNK

                    token = i_token + j_token + k_token

                    if not token in self.vocab:
                        self.vocab[token] = len(self.vocab)

        self.reverse_vocab = {val: key for key, val in self.vocab.items()}

    def encode(self, s: str) -> typing.List:
        if len(s) < self.n:
            s = str(list(s) + [PAD] * (self.n - len(s)))

        result = []
        for i in range(len(s) - self.n + 1):
            substring = s[i:i + self.n]

            if substring in self.vocab:
                result.append(self.vocab[substring])
            else:
                result.append(-1)

        return result

    def decode(self, l: typing.List) -> str:
        s = ""

        for idx in range(len(l)):
            substring = self.reverse_vocab[l[idx]]
            s += substring[0]

        s = s[:-1] + self.reverse_vocab[l[len(l) - 1]]

        return s


class MinimalNGramTokenizer():
    def __init__(self, charset, vocab, n: int = 3):
            self.chardict = {token: idx for idx, token in enumerate(charset)}

            self.vocab = vocab
            self.reverse_vocab = {val: key for key, val in self.vocab.items()}

            self.n = n

    def encode(self, s: str) -> typing.List:
        if len(s) < self.n:
            substring = "".join(list(s) + [PAD] * (self.n - len(s)))
            if substring not in self.vocab:
                self.reverse_vocab[len(self.vocab)] = substring
                self.vocab[substring] = len(self.vocab)

            return [self.vocab[substring]]

        result = []
        for i in range(len(s) - self.n + 1):
            substring = s[i:i+self.n]
            substring = str("".join([c if c.isascii() else UNK for c in substring]))

            if substring not in self.vocab:
                self.reverse_vocab[len(self.vocab)] = substring
                self.vocab[substring] = len(self.vocab)


        return result

    def decode(self, l: typing.List) -> str:
        s = ""

        for idx in range(len(l)):
            substring = self.reverse_vocab[l[idx]]
            substring = self.split_string(substring)
            if not substring[0] == PAD:
                s += substring[0]

        s = s[:-1] + self.reverse_vocab[l[len(l) - 1]]

        return s

    @staticmethod
    def split_string(s: str) -> typing.List:
        if "<" not in s:
            return list(s)

        split_str = s.split("<")

        for i in range(1, len(split_str)):
            split_str[i] = "<" + split_str[i]

        if split_str[0] == "":
            split_str[1:] = split_str[1:]

        return split_str


# A dataset is a container object for the actual data
class JSONDataset(Dataset):
    """Reads a CODWOE JSON dataset"""

    def __init__(
        self,
        file,
        tokenizer=None,
        vocab=None,
        freeze_vocab=False,
        maxlen=256
    ):
        """
        Construct a torch.utils.data.Dataset compatible with torch data API and
        codwoe data.
        args: `file` the path to the dataset file
              `tokenizer` the type of ngram tokenizer to use
              `vocab` a dictionary mapping strings to indices
              `freeze_vocab` whether to update vocabulary, or just replace unknown items with OOV token
              `maxlen` the maximum number of tokens per gloss
        """
        if vocab is None:
            self.vocab = defaultdict(count().__next__)
        else:
            self.vocab = defaultdict(count(len(vocab)).__next__)
            self.vocab.update(vocab)
        pad, eos, bos, unk = (
            self.vocab[PAD],
            self.vocab[EOS],
            self.vocab[BOS],
            self.vocab[UNK],
        )
        if freeze_vocab:
            self.vocab = dict(vocab)

        self.charset = set()
        with open(file, "r") as istr:
            self.items = json.load(istr)
            for gls in (j["gloss"] for j in self.items):
                self.charset.update(gls)

        if tokenizer == "comp":
            self.tokenizer = ComprehensiveNGramTokenizer(self.charset, self.vocab)
        elif tokenizer == "ascii":
            self.tokenizer = AsciiNGramTokenizer(self.charset, self.vocab)
        elif tokenizer == "min":
            self.tokenizer = MinimalNGramTokenizer(self.charset, self.vocab)
        else:
            self.tokenizer = tokenizer

        for json_dict in self.items:
            # in definition modeling test datasets, gloss targets are absent
            if "gloss" in json_dict:
                json_dict["gloss_tensor"] = torch.tensor(
                    [bos] + self.tokenizer.encode(json_dict["gloss"]) + [eos]
                )

                if maxlen:
                    json_dict["gloss_tensor"] = json_dict["gloss_tensor"][:maxlen]
            # in reverse dictionary test datasets, vector targets are absent
            for arch in SUPPORTED_ARCHS:
                if arch in json_dict:
                    json_dict[f"{arch}_tensor"] = torch.tensor(json_dict[arch])
            if "electra" in json_dict:
                json_dict["electra_tensor"] = torch.tensor(json_dict["electra"])

        self.has_gloss = "gloss" in self.items[0]
        self.has_vecs = SUPPORTED_ARCHS[0] in self.items[0]
        self.has_electra = "electra" in self.items[0]
        self.itos = sorted(self.vocab, key=lambda w: self.vocab[w])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    # we're adding this method to simplify the code in our predictions of
    # glosses
    @torch.no_grad()
    def decode(self, tensor):
        """Convert a sequence of indices (possibly batched) to tokens"""
        if tensor.dim() == 2:
            # we have batched tensors of shape [Seq x Batch]
            decoded = []
            for tensor_ in tensor.t():
                decoded.append(self.decode(tensor_))
            return decoded
        else:
            ids = [i.item() for i in tensor if i != self.vocab[PAD]]
            if self.itos[ids[0]] == BOS: ids = ids[1:]
            if self.itos[ids[-1]] == EOS: ids = ids[:-1]

            return self.tokenizer.decode(ids)

    def save(self, file):
        torch.save(self, file)

    @staticmethod
    def load(file):
        return torch.load(file)


# A sampler allows you to define how to select items from your Dataset. Torch
# provides a number of default Sampler classes
class TokenSampler(Sampler):
    """Produce batches with up to `batch_size` tokens in each batch"""

    def __init__(
        self, dataset, batch_size=150, size_fn=len, drop_last=False, shuffle=True
    ):
        """
        args: `dataset` a torch.utils.data.Dataset (iterable style)
              `batch_size` the maximum number of tokens in a batch
              `size_fn` a callable that yields the number of tokens in a dataset item
              `drop_last` if True and the data can't be divided in exactly the right number of batch, drop the last batch
              `shuffle` if True, shuffle between every iteration
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.size_fn = size_fn
        self._len = None
        self.drop_last = drop_last
        self.shuffle = True

    def __iter__(self):
        indices = range(len(self.dataset))
        if self.shuffle:
            indices = list(indices)
            random.shuffle(indices)
        i = 0
        selected = []
        numel = 0
        longest_len = 0
        for i in indices:
            if numel + self.size_fn(self.dataset[i]) > self.batch_size:
                if selected:
                    yield selected
                selected = []
                numel = 0
            numel += self.size_fn(self.dataset[i])
            selected.append(i)
        if selected and not self.drop_last:
            yield selected

    def __len__(self):
        if self._len is None:
            self._len = round(
                sum(self.size_fn(self.dataset[i]) for i in range(len(self.dataset)))
                / self.batch_size
            )
        return self._len


# DataLoaders give access to an iterator over the dataset, using a sampling
# strategy as defined through a Sampler.
def get_dataloader(dataset, batch_size=200, shuffle=False):
    """produce dataloader.
    args: `dataset` a torch.utils.data.Dataset (iterable style)0
          `batch_size` the maximum number of tokens in a batch
          `shuffle` if True, shuffle between every iteration
    """
    # some constants for the closures
    has_gloss = dataset.has_gloss
    has_vecs = dataset.has_vecs
    has_electra = dataset.has_electra
    PAD_idx = dataset.vocab[PAD]

    # the collate function has to convert a list of dataset items into a batch
    def do_collate(json_dicts):
        """collates example into a dict batch; produces ands pads tensors"""
        batch = defaultdict(list)
        for jdict in json_dicts:
            for key in jdict:
                batch[key].append(jdict[key])
        if has_gloss:
            batch["gloss_tensor"] = pad_sequence(
                batch["gloss_tensor"], padding_value=PAD_idx, batch_first=False
            )
        if has_vecs:
            for arch in SUPPORTED_ARCHS:
                batch[f"{arch}_tensor"] = torch.stack(batch[f"{arch}_tensor"])
        if has_electra:
            batch["electra_tensor"] = torch.stack(batch["electra_tensor"])
        return dict(batch)

    if dataset.has_gloss:
        # we try to keep the amount of gloss tokens roughly constant across all
        # batches.
        def do_size_item(item):
            """retrieve tensor size, so as to batch items per elements"""
            return item["gloss_tensor"].numel()

        return DataLoader(
            dataset,
            collate_fn=do_collate,
            batch_sampler=TokenSampler(
                dataset, batch_size=batch_size, size_fn=do_size_item, shuffle=shuffle
            ),
        )
    else:
        # there's no gloss, hence no gloss tokens, so we use a default batching
        # strategy.
        return DataLoader(
            dataset, collate_fn=do_collate, batch_size=batch_size, shuffle=shuffle
        )


def get_train_dataset(train_file, save_dir, tokenizer):
    if (save_dir / "train_dataset.pt").is_file():
        dataset = JSONDataset.load(save_dir / "train_dataset.pt")
    else:
        dataset = JSONDataset(
            train_file,
            tokenizer
        )
        dataset.save(save_dir / "train_dataset.pt")
    return dataset


def get_dev_dataset(dev_file, save_dir, tokenizer):
    if (save_dir / "dev_dataset.pt").is_file():
        dataset = JSONDataset.load(save_dir / "dev_dataset.pt")
    else:
        dataset = JSONDataset(
            dev_file,
            tokenizer
        )
        dataset.save(save_dir / "dev_dataset.pt")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file", type=pathlib.Path, help="path to the train file"
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("models") / f"revdict-baseline",
        help="where to save model & vocab",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Type of ngram tokenizer to use, either 'comp', 'ascii', or 'min'."
    )
    args = parser.parse_args()

    train_dataset = get_train_dataset(args.train_file, args.save_dir, args.tokenizer)