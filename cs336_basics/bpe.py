from __future__ import annotations

import time

import os
from pickle import HIGHEST_PROTOCOL

from cs336_basics import pretokenization
from cs336_basics.heap import MaxHeap
import concurrent.futures
from collections import Counter, defaultdict
import resource
import pickle
from pathlib import Path

from itertools import pairwise

Token = int
Bigram_Token = tuple[Token, Token]
Sequence_Token = tuple[Token, ...]
Seq_ID = int

class BPE:
    def __init__(self, input_path: str, vocab_size, special_tokens: list[str]=None, num_chunks: int = 64):
        if special_tokens is None:
            special_tokens = ["<|endoftext|>"]
        self.num_chunks = num_chunks
        self.num_workers = min(num_chunks, (os.cpu_count() or 1))
        self.special_tokens: list[str] = special_tokens
        self.vocab_size: int = vocab_size
        self.input_path: str = input_path
        self.heap = MaxHeap()
        self.PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.seq_ids: dict[Seq_ID, Sequence_Token] = {}
        self.seq_counts: Counter[Seq_ID] = Counter()
        self.bigram_counts: Counter[Bigram_Token] = Counter()
        self.bigram_indexes: dict[Bigram_Token, set[Seq_ID]] = defaultdict(set)

        self.save_dir = Path("./vocab_merge")
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.vocab_name, self.merges_name = f"{Path(input_path).stem}_{vocab_size}_vocab.pickle", f"{Path(input_path).stem}_{vocab_size}_merges.pickle"

        self.merges: list[tuple[bytes, bytes]] = []

        self.vocab: dict[int, bytes] =  {
            i: bytes([i]) for i in range(256)
        }

        for j in self.special_tokens:
            self.vocab[len(self.vocab)] = j.encode("utf-8")

    def initial_sequence_count(self) -> None:
        seq_counter: Counter[Sequence_Token] = Counter()

        with open(self.input_path, "rb") as f:
            boundaries = pretokenization.find_chunk_boundaries(f,
               self.num_chunks,
               self.special_tokens[0].encode("utf-8"),
            )
            args = [
                (self.input_path, start, end, self.PATTERN)
                for start, end in pairwise(boundaries)
            ]

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for chunk_counts in executor.map(pretokenization.pretokenize, *zip(*args)):
                seq_counter.update(chunk_counts)

        for seq, freq in seq_counter.items():
            seq_id = len(self.seq_ids)
            self.seq_ids[seq_id] = seq
            self.seq_counts[seq_id] = freq

    def initial_bigram_count(self) -> None:
        if not self.seq_counts:
            raise ValueError("self.seq_counts is empty. Call initial_sequence_count before initial_bigram_count")

        bigram_counts = self.bigram_counts
        bigram_indexes = self.bigram_indexes

        for seq_id, seq in self.seq_ids.items():

            seq_freq = self.seq_counts[seq_id]

            if len(seq) < 2:
                continue

            for pair in pairwise(seq):
                bigram_counts[pair] += seq_freq
                bigram_indexes[pair].add(seq_id)

    def initialize_heap(self) -> None:
        for bigram, count in self.bigram_counts.items():
            self.heap.insert(count=count, token1=self.vocab[bigram[0]], token2=self.vocab[bigram[1]], bigram=bigram)

    def rebuild_heap(self) -> None:
        self.heap = MaxHeap()
        self.initialize_heap()

    def find_max_bigram(self) -> Bigram_Token:
        max_bi = self.heap.extract_root()
        while max_bi[3] not in self.bigram_counts or max_bi[0] != self.bigram_counts[max_bi[3]]:
            max_bi = self.heap.extract_root()
        return max_bi[3]

    def merge_once(self) -> None:
        merge_bigram: tuple[Token, Token] = self.find_max_bigram()
        new_token = self.vocab[merge_bigram[0]] + self.vocab[merge_bigram[1]]

        token_id: int = len(self.vocab)
        self.vocab[token_id] = new_token
        self.merges.append((self.vocab[merge_bigram[0]], self.vocab[merge_bigram[1]]))

        seq_ids =  tuple(self.bigram_indexes[merge_bigram])

        for seq_id in seq_ids:
            if seq_id not in self.seq_counts:
                continue

            old_seq = self.seq_ids[seq_id]
            seq_freq = self.seq_counts[seq_id]

            new_blist: list[Token] = []
            i = 0

            while i < len(old_seq):
                if i < len(old_seq) - 1 and old_seq[i] == merge_bigram[0] and old_seq[i+1] == merge_bigram[1]:
                    new_blist.append(token_id)
                    i+=2
                else:
                    new_blist.append(old_seq[i])
                    i+=1

            new_seq = tuple(new_blist)

            if new_seq == old_seq:
                continue

            old_bi_count = Counter(pairwise(old_seq))
            new_bi_count = Counter(pairwise(new_blist))

            self.seq_ids[seq_id] = new_seq

            for pair in old_bi_count.keys() | new_bi_count.keys():

                old_count = old_bi_count[pair]
                new_count = new_bi_count[pair]
                diff = new_count - old_count

                if diff!=0:
                    self.bigram_counts[pair] += diff * seq_freq
                    updated_count = self.bigram_counts[pair]

                    if updated_count > 0:
                        self.heap.insert(
                            count=updated_count,
                            bigram=pair,
                            token1=self.vocab[pair[0]],
                            token2=self.vocab[pair[1]],
                        )
                    else:
                        self.bigram_counts.pop(pair, None)

                if old_count > 0 and new_count == 0:
                    self.bigram_indexes[pair].discard(seq_id)

                elif old_count == 0 and new_count > 0:
                    self.bigram_indexes[pair].add(seq_id)

            if self.heap.size() > 3 * len(self.bigram_counts):
                self.rebuild_heap()

    def save(self):
        vocab_save = self.save_dir / self.vocab_name
        merges_save = self.save_dir / self.merges_name

        with open(vocab_save, "wb") as f:
            pickle.dump(self.vocab, f, protocol=HIGHEST_PROTOCOL)

        with open(merges_save, "wb") as f:
            pickle.dump(self.merges, f, protocol=HIGHEST_PROTOCOL)

    def train(self):
        self.initial_sequence_count()
        self.initial_bigram_count()
        self.initialize_heap()
        while len(self.vocab) < self.vocab_size and self.bigram_counts != {}:
            self.merge_once()
        self.save()
        return self.vocab, self.merges

if __name__ == "__main__":
    start_time = time.perf_counter()
    Tokenizer = BPE(input_path = "../data/owt_train.txt", vocab_size = 32000)
    vocab, merges = Tokenizer.train()
    print(vocab)
    print(merges)
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    end_time = time.perf_counter()
    print(f" Exec time: {end_time-start_time}")
    print(f"Peak memory usage: {usage}")