import importlib.metadata
import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import time
import multiprocessing as mp
import pickle 
import json
import math
from typing import Iterable, Iterator
# from memory_profiler import profile

__version__ = importlib.metadata.version("cs336_basics")


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESS = 4

def pretokenize_chunk(input_path, start, end, pat, special_tokens):
    pre_tokens = defaultdict(int)
    with open(input_path, 'rb') as f:
            f.seek(start)
            # Binary mode ('rb') preserves raw bytes — no automatic line ending normalization.
            # On Windows, files may have \r\n (carriage return + newline) instead of just \n.
            # The reference snapshot was created on Linux (text mode, auto-normalizes \r\n -> \n),
            # so we must manually normalize here to match. \r\n must be replaced before \r
            # to avoid double-converting \r\n into two \n's.
            text = f.read(end - start).decode(encoding= 'utf-8').replace('\r\n', '\n').replace('\r', '\n')
    parts = [text]
    if special_tokens:
        pattern = "|".join(re.escape(t) for t in special_tokens)
        parts = re.split(pattern, text)
    
    
    for part in parts: 
        for m in re.finditer(pat, part):
            '''
            Note that here the tuple() operation converts bytes to integers.
            This is actually desired because bytes can't exceed 255.
            So after merges, you can't store bytes([256]) in the key
            '''
            pre_tokens[tuple(m.group().encode(encoding='utf-8'))] += 1
    return pre_tokens

class BPE():
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)} # dict[id, bytes]
        self.new_id = 256
        self.merges = []
        self.pre_tokens = defaultdict(int) # dict[tuple(id values), count] 
        self.pair_index = defaultdict(set) # A helper mapping dict[pair, set[pre_token_sequence]] that tells me "if I merge this pair, which pre-tokens will be affected" 
        self.pairs = defaultdict(int) # dict(tuple(ids), count), maintains the count of each pair of values
    
    def add_special_tokens(self, special_tokens: list[str]):
        for s in special_tokens:
            self.vocab[self.new_id] = s.encode('utf-8') ## Remember to always convert to bytes
            self.new_id += 1
    
    def pre_tokenization(self, input_path, pat = PAT, special_tokens = None):
        with open(input_path, 'rb') as f:
            chunk_boundaries = self.find_chunk_boundaries(f, NUM_PROCESS, b"<|endoftext|>")
        with mp.Pool(processes=NUM_PROCESS) as pool:
            inputs = [(input_path, start, end, pat, special_tokens) for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]
            results = pool.starmap(pretokenize_chunk, inputs)
        
        for result in results:
            # print(result)
            for key, value in result.items():
                self.pre_tokens[key] += value
        
        
    # @profile
    def train_bpe(self, input_path, vocab_size, special_tokens):
        # with open(input_path, encoding='utf-8') as f:
        #     file = f.read()
        self.add_special_tokens(special_tokens)
        self.pre_tokenization(input_path, special_tokens= special_tokens)
        
        for k in self.pre_tokens.keys():
            for i in range(len(k) -1):
                self.pairs[k[i:i+2]] += self.pre_tokens[k]
                self.pair_index[k[i:i+2]].add(k)
                
        while len(self.vocab) < vocab_size:
            max_key = max(self.pairs, key= lambda x: (self.pairs[x], self.vocab[x[0]], self.vocab[x[1]]))
            # print(max_key,self.pairs[max_key])
            self.merges.append((self.vocab[max_key[0]], self.vocab[max_key[1]]))
            self.vocab[self.new_id] = self.vocab[max_key[0]] + self.vocab[max_key[1]]
            affected_pretokens = set(self.pair_index[max_key]) #Makes a copy 
            
            
            # new_pre_tokens = defaultdict(int)
            for k in affected_pretokens:
                new_key = []
                i = 0
                while i < len(k):
                    if i < len(k) - 1 and k[i:i+2] == max_key:
                        new_key.append(self.new_id)
                        if i > 0:
                            self.pairs[(new_key[-2], k[i])] -= self.pre_tokens[k]
                            self.pairs[tuple(new_key[-2:])] += self.pre_tokens[k]

                        if i < len(k) - 2:
                            self.pairs[k[i+1:i+3]] -= self.pre_tokens[k]
                            self.pairs[(self.new_id, k[i+2])] += self.pre_tokens[k]
                        i += 2
                    else:
                        new_key.append(k[i])
                        i += 1

                # Remove k from pair_index for all pairs it contained
                for j in range(len(k) - 1):
                    self.pair_index[k[j:j+2]].discard(k)

                self.pre_tokens[tuple(new_key)] = self.pre_tokens.pop(k)

                # Add new_key to pair_index for all pairs it now contains
                new_key_tuple = tuple(new_key)
                for j in range(len(new_key) - 1):
                    self.pair_index[new_key_tuple[j:j+2]].add(new_key_tuple)
                
                
            self.pairs[max_key] = 0
            # self.pre_tokens = new_pre_tokens
            self.new_id += 1
            # print("self.pretokens", self.pre_tokens)
        
        return self.vocab, self.merges
                
                    
        
        
    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))


class Tokenizer():
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reverse_merges = {merges[i]: i for i in range(len(merges))}
        self.reverse_vocab = {vocab[i]: i for i in range(len(vocab))}
        
        
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        """
        cls = the Tokenizer class when used in a classmethod
        self = an instance of Tokenizer
        """
        with open(vocab_filepath, 'r') as f:
            vocab = json.load(f)
            ### When you read from json, both keys and values are strings, need to convert to int: bytes
            vocab = {int(k): v.encode('latin-1') for k, v in vocab.items()}
        
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)
    
    
    def merge_pre_token(self, pt):
        while True:
            min_pos = math.inf
            track = 0
            for i in range(len(pt)-1):
                ind = self.reverse_merges.get((self.vocab[pt[i]], self.vocab[pt[i+1]]), math.inf)
                if ind < min_pos:
                    track = i
                    min_pos = ind
            if min_pos == math.inf:
                break
            new_vocab = self.reverse_vocab[self.vocab[pt[track]] + self.vocab[pt[track + 1]]]
            pt = pt[:track] + tuple([new_vocab]) + pt[track+2:]
        
        return pt   
            
                    
    def encode(self, text: str) -> list[int]:
        parts = [text]
        if self.special_tokens:
            pattern = "|".join(re.escape(t) for t in sorted(self.special_tokens, key= lambda x: len(x), reverse= True))
            parts = re.split(f"({pattern})", text) # Note here need to use f-string 
    
        pre_tokens = []
        for part in parts:
            if self.special_tokens and part in self.special_tokens:
                pre_tokens.append(part)
                continue
            for m in re.finditer(PAT, part):
                pre_tokens.append(tuple([self.reverse_vocab[bytes([b])] for b in m.group().encode(encoding='utf-8')]))
        
        res = []
        for pt in pre_tokens:
            if isinstance(pt, tuple):
                temp = self.merge_pre_token(pt)
                for b in temp:
                    res.append(b)
            
            else:
                res.append(self.reverse_vocab[pt.encode(encoding='utf-8')])
        
        return res


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        res = []
        for id in ids:
            res.append(self.vocab[id])
        
        return b''.join(res).decode('utf-8', errors='replace')

# if __name__ == '__main__':
    # try:
    #     bpe = BPE()
    #     t1 = time.time()
    #     # print(bpe.train_bpe('../data/mini.txt', 260, []))
    #     vocab, merges = bpe.train_bpe('../data/owt_train.txt', 32000, ["<|endoftext|>"])
    #     with open('owt_merges.pkl', 'wb') as f:
    #         pickle.dump(merges, f)

    #     with open('owt_vocab.json', 'w') as f:
    #         json.dump({str(k): v.decode('latin-1') for k, v in vocab.items()}, f)

    #     # print(merges)s
    #     print(time.time() - t1)
    # except KeyboardInterrupt:
    #     print("Interrupted")

    # import random

    # SPECIAL_TOKENS = ["<|endoftext|>"]
    # owt_tokenizer = Tokenizer.from_files('owt_vocab.json', 'owt_merges.pkl', SPECIAL_TOKENS)
    # ts_tokenizer  = Tokenizer.from_files('tiny_stories_vocab.json', 'tiny_stories_merges.pkl', SPECIAL_TOKENS)

    # def sample_documents(path, n=10, seed=42):
    #     with open(path, encoding='utf-8') as f:
    #         text = f.read()
    #     docs = [d.strip() for d in text.split('<|endoftext|>') if d.strip()]
    #     random.seed(seed)
    #     return random.sample(docs, min(n, len(docs)))

    # owt_docs = sample_documents('../data/owt_train.txt')
    # ts_docs  = sample_documents('../data/TinyStories.txt')

    # def compression_ratio(tokenizer, docs):
    #     total_bytes = sum(len(d.encode('utf-8')) for d in docs)
    #     total_tokens = sum(len(tokenizer.encode(d)) for d in docs)
    #     return total_bytes / total_tokens  # bytes per token

    # print("=== Compression Ratio (bytes per token) ===")
    # print(f"OWT tokenizer  on OWT docs:        {compression_ratio(owt_tokenizer, owt_docs):.2f}")
    # print(f"OWT tokenizer  on TinyStories docs: {compression_ratio(owt_tokenizer, ts_docs):.2f}")
    # print(f"TS tokenizer   on TinyStories docs: {compression_ratio(ts_tokenizer, ts_docs):.2f}")
    # print(f"TS tokenizer   on OWT docs:         {compression_ratio(ts_tokenizer, owt_docs):.2f}")

    # print("\n=== Throughput (tokens/sec) ===")
    # all_docs = owt_docs + ts_docs
    # combined_text = ' '.join(all_docs)
    # t0 = time.time()
    # tokens = owt_tokenizer.encode(combined_text)
    # elapsed = time.time() - t0
    # print(f"OWT tokenizer: {len(tokens)/elapsed:,.0f} tokens/sec  ({len(tokens)} tokens in {elapsed:.2f}s)")
    
    