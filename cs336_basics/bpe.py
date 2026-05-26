from __future__ import annotations

import time

from cs336_basics import pretokenization
from cs336_basics import heap
import regex as re
import concurrent.futures
from collections import Counter
import resource
import pickle
from pathlib import Path

Token = int
Bigram_Token = tuple[Token, Token]
Sequence_Token = tuple[Token, ...]

class BPE:
    def __init__(self, input_path: str, vocab_size, special_tokens: list[str]=None, num_chunks: int=6):
        if special_tokens is None:
            special_tokens = ["<|endoftext|>"]
        self.special_tokens: list[str] = special_tokens
        self.vocab_size: int = vocab_size
        self.input_path: str = input_path
        self.num_chunks = num_chunks
        self.heap = heap.MaxHeap()
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.seq_counts: dict[Sequence_Token, int] = {}
        self.bigram_counts: dict[Bigram_Token, int] = {}
        self.bigram_indexes: dict[Bigram_Token, set[Sequence_Token]] = {}

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
        with open(self.input_path, "rb") as f:
            boundaries = pretokenization.find_chunk_boundaries(f, self.num_chunks, self.special_tokens[0].encode("utf-8"))
            args = [(self.input_path, start, end, self.PAT) for start, end in zip(boundaries[:-1], boundaries[1:])]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            chunks_counts = executor.map(pretokenization.pretokenize, *zip(*args))
        for chunk_counts in chunks_counts:
            for seq, seq_freq in chunk_counts.items():
                self.seq_counts[seq] = self.seq_counts.get(seq, 0) + seq_freq


    def initial_bigram_count(self) -> None:
        if not self.seq_counts:
            raise ValueError("self.seq_counts is empty. Call initial_sequence_count before initial_bigram_count")
        for seq, seq_freq in self.seq_counts.items():
            if len(seq) < 2:
                continue
            for pair in zip(seq[:-1], seq[1:]):
                self.bigram_counts[pair] = self.bigram_counts.get(pair, 0) + seq_freq
                inner = self.bigram_indexes.setdefault(pair, set())
                inner.add(seq)

    def initialize_heap(self) -> None:
        for bigram, count in self.bigram_counts.items():
            self.heap.insert(count=count, token1=self.vocab[bigram[0]], token2=self.vocab[bigram[1]], bigram=bigram)

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

        seqs =  self.bigram_indexes[merge_bigram].copy()
        for old_seq in seqs:
            new_blist: list[Token] = []
            i = 0
            seq_freq = self.seq_counts[old_seq]
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

            old_bi_count = Counter(zip(old_seq[:-1], old_seq[1:]))
            new_bi_count = Counter(zip(new_blist[:-1], new_blist[1:]))
            self.seq_counts[new_seq] = self.seq_counts.setdefault(new_seq, 0) + seq_freq

            for pair in old_bi_count | new_bi_count:
                diff = new_bi_count[pair] - old_bi_count[pair]

                if pair not in self.bigram_indexes:
                    self.bigram_indexes[pair] = self.bigram_indexes.setdefault(pair, set())
                if pair not in self.bigram_counts:
                    self.bigram_counts[pair] = self.bigram_counts.setdefault(pair, 0)

                if diff == 0:
                    inner = self.bigram_indexes[pair]
                    inner.discard(old_seq)
                    inner.add(new_seq)

                elif diff < 0:
                    self.bigram_counts[pair] += diff * seq_freq
                    self.heap.insert(count=self.bigram_counts[pair], token1=self.vocab[pair[0]], token2=self.vocab[pair[1]], bigram=pair)

                    inner = self.bigram_indexes[pair]
                    inner.discard(old_seq)

                    if new_bi_count[pair] > 0:
                        inner = self.bigram_indexes[pair]
                        inner.add(new_seq)

                else:
                    self.bigram_counts[pair] += diff * seq_freq
                    self.heap.insert(count=self.bigram_counts[pair], token1=self.vocab[pair[0]], token2=self.vocab[pair[1]], bigram=pair)
                    inner = self.bigram_indexes[pair]
                    inner.add(new_seq)

                if self.bigram_counts[pair] <= 0:
                    self.bigram_counts.pop(pair, None)
                if not self.bigram_indexes[pair]:
                    self.bigram_indexes.pop(pair, None)

            self.seq_counts.pop(old_seq, None)

    def save(self):
        vocab_save = self.save_dir / self.vocab_name
        merges_save = self.save_dir / self.merges_name
        vocab_save.write_bytes(pickle.dumps(self.vocab))
        merges_save.write_bytes(pickle.dumps(self.merges))

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