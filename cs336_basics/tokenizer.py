from __future__ import annotations
import pickle
from typing import Iterable, Iterator

import regex as re
import concurrent.futures
import time

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges

        self.merges_rank = {self.merges[i]: i for i in range(len(self.merges))}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.pretoken_cache = {}

        if special_tokens is None:
            special_tokens = ["<|endoftext|>"]
        self.special_tokens = set(special_tokens)
        for special_token in self.special_tokens:
            if special_token.encode() not in self.reverse_vocab:
                self.vocab[len(self.vocab)] = special_token.encode()

        self.special_regex = "|".join(sorted([re.escape(token) for token in self.special_tokens], key=len, reverse=True))
        self.special_pat = re.compile(f"({self.special_regex})")
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab = vocab, merges = merges, special_tokens=special_tokens)

    def merge(self, pretoken) -> list[int]:

        if pretoken in self.pretoken_cache:
            return self.pretoken_cache[pretoken]

        flag = True
        b_pretoken = pretoken.encode()
        bytearr = [b_pretoken[i:i+1] for i in range(len(b_pretoken))]
        max_val = len(self.merges_rank)
        bigram_seq = zip(bytearr[:-1], bytearr[1:])
        while flag:
            flag = False
            min_so_far = (max_val, b"", 0)
            for i, bg in enumerate(bigram_seq):
                bi_rank = self.merges_rank.get(bg, max_val)
                if bi_rank < min_so_far[0]:
                    min_so_far = (bi_rank, bg[0]+bg[1], i)
                    flag = True
            if flag:
                bytearr[min_so_far[2]] = min_so_far[1]
                bytearr.pop(min_so_far[2]+1)
                bigram_seq = zip(bytearr[:-1], bytearr[1:])
        id_seq = [self.reverse_vocab[token] for token in bytearr]
        self.pretoken_cache[pretoken] = id_seq
        return id_seq


    def encode(self, text):
        pieces = re.split(self.special_pat, text)
        id_seq = []
        for i in pieces:
            if i in self.special_tokens:
                id_seq.append(self.reverse_vocab[i.encode()])
            else:
                for j in re.findall(self.PAT, i):
                    id_seq.extend(self.merge(j))
        return id_seq

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            pieces = re.split(self.special_pat, text_chunk)
            for i in pieces:
                if i in self.special_tokens:
                    yield self.reverse_vocab[i.encode()]
                else:
                    for j in re.findall(self.PAT, i):
                        for token_id in self.merge(j):
                            yield token_id

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[idx] for idx in ids).decode(errors="replace")


if __name__ == "__main__":
    text = "Assalomu aleykum"
    tokenizer = Tokenizer.from_files("./vocab_merge/TinyStoriesV2-GPT4-train_10000_vocab.pickle", "./vocab_merge/TinyStoriesV2-GPT4-train_10000_merges.pickle", special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    encode_out = tokenizer.encode(text)
    print(encode_out)
    decode_out = tokenizer.decode(encode_out)
    print(decode_out)

