import importlib.metadata
import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import time
import multiprocessing as mp

__version__ = importlib.metadata.version("cs336_basics")


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESS = 12

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
            Note that here the keys to the pre_tokens are tuples of integers
            This is actually desired because bytes can't exceed 255.
            So after merges, you can't store bytes([256]) in the key
            '''
            pre_tokens[tuple(m.group().encode(encoding='utf-8'))] += 1
    return pre_tokens

class BPE():
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.new_id = 256
        self.merges = []
        self.pre_tokens = defaultdict(int)
    
    
    def add_special_tokens(self, special_tokens: list[str]):
        for s in special_tokens:
            self.vocab[self.new_id] = s.encode('utf-8') ## Remember to always convert to bytes
            self.new_id += 1
    
    def pre_tokenization(self, input_path, pat = PAT, special_tokens = []):
        with open(input_path, 'rb') as f:
            chunk_boundaries = self.find_chunk_boundaries(f, NUM_PROCESS, b"<|endoftext|>")
        with mp.Pool(processes=NUM_PROCESS) as pool:
            inputs = [(input_path, start, end, pat, special_tokens) for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]
            results = pool.starmap(pretokenize_chunk, inputs)
        
        for result in results:
            # print(result)
            for key, value in result.items():
                self.pre_tokens[key] += value
        
        
    
    def train_bpe(self, input_path, vocab_size, special_tokens):
        # with open(input_path, encoding='utf-8') as f:
        #     file = f.read()
        self.add_special_tokens(special_tokens)
        self.pre_tokenization(input_path, special_tokens= special_tokens)
        while len(self.vocab) < vocab_size:
            self.pairs = defaultdict(int)
            for k in self.pre_tokens.keys():
                for i in range(len(k) -1):
                    self.pairs[k[i:i+2]] += self.pre_tokens[k]
                    
            
            max_key = max(self.pairs, key= lambda x: (self.pairs[x], self.vocab[x[0]], self.vocab[x[1]]))
            # print(max_key,self.pairs[max_key])
            self.merges.append((self.vocab[max_key[0]], self.vocab[max_key[1]]))
            self.vocab[self.new_id] = self.vocab[max_key[0]] + self.vocab[max_key[1]]
            
            new_pre_tokens = defaultdict(int)
            for k in self.pre_tokens.keys():
                new_key = []
                i = 0
                while i < len(k):
                    if i < len(k) - 1 and k[i:i+2] == max_key:
                        new_key.append(self.new_id)
                        i += 2
                    
                    else:
                        new_key.append(k[i])
                        i += 1
                
                new_pre_tokens[tuple(new_key)] = self.pre_tokens[k]
            
            self.pre_tokens = new_pre_tokens
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


if __name__ == '__main__':
    try:
        bpe = BPE()
        t1 = time.time()
        # print(bpe.train_bpe('../data/mini.txt', 260, []))
        vocab, merges = bpe.train_bpe('../data/TinyStories.txt', 10000, ["<|endoftext|>"])
        print(merges)
        print(time.time() - t1)
    except KeyboardInterrupt:
        print("Interrupted")