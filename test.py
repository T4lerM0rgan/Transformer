# from __future__ import annotations
#
# import os
# from collections.abc import Iterable
# from typing import IO, Any, BinaryIO
#
# import numpy.typing as npt
# import torch
# from jaxtyping import Bool, Float, Int
# from sympy.printing.pretty.pretty_symbology import vobj
# from torch import Tensor
# import multiprocessing as mp
# import regex as re
# from torch.fx.experimental.unification.core import seq
#
# from cs336_basics import pretokenization_example
# import concurrent.futures
#
# from tests.conftest import vocab_size
#
# #
# # def run_train_bpe(
# #     input_path: str | os.PathLike,
# #     vocab_size: int,
# #     special_tokens: list[str],
# #     **kwargs,
# # ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
# #
# #     global_seq_counts = {}
# #     global_bi_counts = {}
# #     global_bi_indexes = {}
# #
# #     with open(input_path, "rb") as f: #split the file into chunks of text to parallelize
# #         num_processes = 16
# #         chunks = []
# #         boundaries = pretokenization_example.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
# #         for start, end, in zip(boundaries[:-1], boundaries[1:]):
# #             f.seek(start)
# #             chunks.append(f.read(end-start))
# #         for i in range(len(chunks)):
# #             chunks[i] = chunks[i].decode("utf-8")
# #
# #     with concurrent.futures.ProcessPoolExecutor() as executor: #pretokenize the chunks
# #         chunks_counts = executor.map(pretokenization_example.process_file, chunks)
# #         for chunk_counts in chunks_counts:
# #             for seq in chunk_counts:
# #                 global_seq_counts[seq] = global_seq_counts.get(seq,0) + chunk_counts[seq]
# #                 for pair in zip(seq[:-1], seq[1:]):
# #                     global_bi_counts[pair] = global_bi_counts.get(pair, 0) + chunk_counts[seq]
# #                     tmp_set = global_bi_indexes.get(pair, set())
# #                     tmp_set.add(seq)
# #                     global_bi_indexes[pair] = tmp_set
# #     vocab = {
# #         i: bytes([i]) for i in range(256),
# #     }
# #     for j in special_tokens:
# #         vocab[len(vocab)] = j.encode("utf-8")
# #
# #     merges = []
# #
# #     while True:
# #         max_bi = max(global_bi_counts, key=global_bi_counts.get)
# #         new_tkn = vocab[max_bi[0]] + vocab[max_bi[1]]
# #         token_id = len(vocab)
# #         vocab[token_id] = new_tkn
# #         merges.append(max_bi)
# #         seqs = global_bi_indexes[max_bi]
# #         for seq in seqs:
# #             merge_merge = False
# #             blist = list(seq)
# #             new_blist = []
# #             i = 0
# #             bi_freq = global_seq_counts[seq]
# #             while i < len(blist):
# #                 if i < len(blist) - 1 and blist[i] == max_bi[0] and blist[i+1] == max_bi[1]:
# #                     new_blist.append(token_id)
# #                     try:
# #                         if merge_merge:
# #                             pass
# #                         else:
# #                             chng_bi = tuple(blist[i-1:i+1])
# #                             global_bi_counts[chng_bi] -= bi_freq
# #                             global_bi_indexes[chng_bi][seq] -= 1
# #                             if global_bi_indexes[chng_bi][seq] == 0:
# #                                 global_bi_indexes[chng_bi].pop(seq)
# #                     except:
# #                         pass
# #                     try:
# #                         if (blist[i+2], blist[i+3]) == max_bi:
# #                             merge_merge = True
# #                         else:
# #                             merge_merge = False
# #                         chng_bi = tuple(blist[i + 1:i + 3])
# #                         global_bi_counts[chng_bi] -= global_seq_counts[seq]
# #                         global_bi_indexes[chng_bi][seq] -= 1
# #                         if global_bi_indexes[chng_bi][seq] == 0:
# #                             global_bi_indexes[chng_bi].pop(seq)
# #                     except:
# #                         pass
# #                     i += 2
# #                 else:
# #                     new_blist.append(blist[i])
# #                     i += 1
# #
# #             new_seq = tuple(new_blist)
# #             i = 0
# #             while i < len(new_seq):
# #
# #             global_seq_counts[new_seq] = global_seq_counts.get(new_seq, 0) + global_seq_counts[seq]
# #             global_seq_counts.pop(seq)
# #
# #         global_bi_indexes.pop(max_bi)
# #         global_bi_counts.pop(max_bi)
# #
# #         if len(vocab) == vocab_size:
# #             break
# max_bi = (119, 101)
# blist = list((110, 101, 119, 101, 119, 101, 101, 119, 111, 115, 116))
# new_blist = []
# i = 0
# while i < len(blist):
#     if i < len(blist) - 1 and blist[i] == max_bi[0] and blist[i + 1] == max_bi[1]:
#         new_blist.append(256)
#         i += 2
#     else:
#         new_blist.append(blist[i])
#         i += 1
#
# # if pair not in self.bigram_indexes:
# #     self.bigram_indexes[pair] = self.bigram_indexes.setdefault(pair, {})
# # if pair not in self.bigram_counts:
# #     self.bigram_counts[pair] = self.bigram_counts.setdefault(pair, 0)
#
#         def pretokenize(
#                 chunk: str,
#         ):
#             PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
#             special_tokens = [re.escape(i) for i in ["<|endoftext|>"]]
#             counts = {}
#             rm_st = re.split("|".join(special_tokens), chunk)
#             for sequence in rm_st:
#                 for i in re.finditer(PAT, sequence):
#                     pretoken = tuple(i.group().encode("utf-8"))
#                     counts[pretoken] = counts.get(pretoken, 0) + 1
#             return counts

import pickle

with open("./cs336_basics/vocab_merge/TinyStoriesV2-GPT4-train_10000_merges.pickle", "rb") as f:
    data = pickle.load(f)
print(data)

