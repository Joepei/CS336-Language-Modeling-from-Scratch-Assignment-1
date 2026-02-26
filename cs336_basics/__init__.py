import importlib.metadata
import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import time

__version__ = importlib.metadata.version("cs336_basics")


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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
    
    def pre_tokenization(self, text, pat = PAT, special_tokens = []):
        
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
                self.pre_tokens[tuple(m.group().encode(encoding='utf-8'))] += 1
        
        
        
    
    def train_bpe(self, input_path, vocab_size, special_tokens):
        with open(input_path, encoding='utf-8') as f:
            file = f.read()
        self.add_special_tokens(special_tokens)
        self.pre_tokenization(text = file, special_tokens= special_tokens)
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



bpe = BPE()
t1 = time.time()
# print(bpe.train_bpe('../data/mini.txt', 260, []))
vocab, merges = bpe.train_bpe('../data/TinyStories.txt', 10, ["<|endoftext|>"])
print(time.time() - t1)