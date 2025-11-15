import multiprocessing
import os
import pickle
from collections import Counter, defaultdict, UserDict
from typing import BinaryIO
import regex as re
import tqdm
from tqdm import tqdm
from typing import Iterable, Iterator
import pathlib

ROOT = pathlib.Path("/Users/chengze/work/assignment1-basics")
# ROOT = pathlib.Path("/home/chengze/work/336")

class Vocabulary(UserDict):
    def __init__(self):
        super().__init__()

    def add(self, token: bytes):
        numel = len(self.data)
        self.data[token] = numel


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes] = {},
                 merges: list[tuple[bytes, bytes]] = [],
                 special_tokens: list[str] | None = None):
        self.num_processes = 16
        self.read_from_checkpoint = False
        self.write_to_checkpoint = False
        self.i2b = vocab
        self.b2i = {v: k for k, v in vocab.items()}
        self.merges = merges
        # lower index means higher priority
        self.merge_priority = {merges[i]: -i for i in range(len(merges))}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        self.special_tokens_bytes = {x.encode('UTF-8') for x in self.special_tokens}


    @staticmethod
    def build_pre_tokens(text: str, special_tokens: list[str] | None = None, preserve_special_tokens = False) -> list: 
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        special_tokens = special_tokens or []
        if len(special_tokens) > 0:
            pattern = "|".join(map(re.escape, special_tokens))
            if preserve_special_tokens:
                pattern = "(" + pattern + ")"
            docs = re.split(pattern, text)
        else:
            docs = [text]

        pre_tokens = []
        for doc in docs:
            if doc in special_tokens:
                pre_tokens.append(doc.encode("UTF-8"))
                continue
            pre_tokens_doc = re.findall(PAT, doc)
            pre_tokens.extend(
                [
                    tuple(
                        token.encode("UTF-8")[i: i + 1]
                        for i in range(len(token.encode("UTF-8")))
                    )
                    for token in pre_tokens_doc
                ]
            )
        return pre_tokens
    
    @staticmethod
    def ThreadWorkPreTokenization(start, end, input_path, special_tokens):
        """Build pre-tokens for a chunk of the file."""
        with open(input_path, "rb") as f:
            f.seek(start)
            text = f.read(end - start).decode("utf-8")
        pre_tokens = BPETokenizer.build_pre_tokens(text, special_tokens)
        return Counter(pre_tokens)

    @staticmethod
    def find_chunk_boundaries(
        file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(
            split_special_token, bytes
        ), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [
            i * chunk_size for i in range(desired_num_chunks + 1)]
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

    @staticmethod
    def count_pairs(token_counter):
        byte_pair_counts = defaultdict(int)
        for pre_token, pre_token_count in token_counter.items():
            for x in range(len(pre_token) - 1):
                # Count all pairs in all tokens
                byte_pair_counts[
                    (pre_token[x], pre_token[x + 1])
                ] += pre_token_count
        return byte_pair_counts

    @staticmethod
    def merge_tokens(tokens, merged):
        idx = set()
        for x in range(len(tokens) - 1):
            if (tokens[x], tokens[x + 1]) == merged:
                idx.add(x)       
        if not idx:
            return tokens, idx
        
        new_tokens = []
        id = 0
        while id < len(tokens):
            if id in idx:
                new_tokens.append(merged[0] + merged[1])
                id += 2
            else:
                new_tokens.append(tokens[id])
                id += 1
        return tuple(new_tokens), idx

    # @profile
    def train(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
    ):
        vocab = Vocabulary()
        merges = []

        for x in range(256):
            vocab.add(x.to_bytes())
        for x in special_tokens:
            vocab.add(x.encode("UTF-8"))
        
        # Pre-tokenization
        if self.read_from_checkpoint:
            with open(ROOT / "checkpoints/pretokens.pkl", "rb") as f:
                res = pickle.load(f)
                token_counter = res["token_counter"]
                byte_pair_counts = res["byte_pair_counts"]
        else:
            token_counter = Counter()
            with open(input_path, "rb") as f:
                boundaries = BPETokenizer.find_chunk_boundaries(
                    f, self.num_processes, "<|endoftext|>".encode("utf-8")
                )
                # Create arguments for each worker process
                chunk_args = [
                    (start, end, input_path, special_tokens)
                    for start, end in zip(boundaries[:-1], boundaries[1:])
                ]
                with multiprocessing.Pool() as pool:
                    results = pool.starmap(
                        BPETokenizer.ThreadWorkPreTokenization, chunk_args)
            for res in results:
                token_counter += res

            byte_pair_counts = BPETokenizer.count_pairs(
                token_counter)
            if self.write_to_checkpoint:
                with open(ROOT / "checkpoints/pretokens.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "token_counter": token_counter,
                            "byte_pair_counts": byte_pair_counts,
                        },
                        f,
                    )
        # Merge
        for _ in tqdm(range(vocab_size - len(vocab))):
            if len(byte_pair_counts) == 0:
                print("No more byte pairs to merge.")
                # If no byte pairs left, we can stop early
                break

            highest_freq = max(byte_pair_counts.values())
            merged = None
            for k, v in byte_pair_counts.items():
                if v != highest_freq:
                    continue
                if not merged:
                    merged = k
                else:
                    merged = max(merged, k)
            # We have a winner, we need add it to vocab and update the tokens
            merges.append(merged)
            new_token = merged[0] + merged[1]
            vocab.add(new_token)

            new_token_counter = {}
            for pre_token, pre_token_count in token_counter.items():
                # tokens, merge_indices = BPETokenizer.merge_tokens(pre_token, merged)
                idx = set()
                for x in range(len(pre_token) - 1):
                    if (pre_token[x], pre_token[x + 1]) == merged:
                        idx.add(x)     
                if not idx:
                    new_token_counter[pre_token] = pre_token_count 
                    continue

                new_tokens = []
                id = 0
                while id < len(pre_token):
                    if id in idx:
                        new_tokens.append(merged[0] + merged[1])
                        id += 2
                    else:
                        new_tokens.append(pre_token[id])
                        id += 1

                new_token_counter[tuple(new_tokens)] = pre_token_count
                for id in idx:
                    if id > 0:
                        byte_pair_counts[
                            (pre_token[id - 1], merged[0])
                        ] -= pre_token_count
                        if byte_pair_counts[(pre_token[id - 1], merged[0])] == 0:
                            byte_pair_counts.pop(
                                (pre_token[id - 1], merged[0]), None)
                        byte_pair_counts[
                            (pre_token[id - 1], new_token)
                        ] += pre_token_count
                    if id < len(pre_token) - 2:
                        byte_pair_counts[
                            (merged[1], pre_token[id + 2])
                        ] -= pre_token_count
                        if byte_pair_counts[(merged[1], pre_token[id + 2])] == 0:
                            byte_pair_counts.pop(
                                (merged[1], pre_token[id + 2]), None)
                        byte_pair_counts[
                            (new_token, pre_token[id + 2])
                        ] += pre_token_count

            token_counter = new_token_counter
            byte_pair_counts.pop(merged, None)

        self.vocab = vocab
        self.merges = merges
        reversed_voc = {v: k for k, v in vocab.items()}
        return reversed_voc, merges

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def decode(self, ids: list[int]) -> str:
        textbytes = b''
        for id in ids:
            textbytes += self.i2b[id]
        return textbytes.decode("utf-8", "replace")

    def encode(self, text: str) -> list[int]:
        tokens = BPETokenizer.build_pre_tokens(text, self.special_tokens, preserve_special_tokens=True)
        res = []
        for token in tokens:
            if token in self.special_tokens_bytes:
                res.append(self.b2i[token])
            else:
                while True:
                    valid_pairs = []
                    for i in range(len(token) - 1):
                        pair = (token[i], token[i+1])
                        if pair in self.merge_priority:
                            valid_pairs.append((self.merge_priority[pair], pair))
                    if not valid_pairs:
                        break
                    _, merge = max(valid_pairs)
                    token, _ = BPETokenizer.merge_tokens(token, merge)
                res.extend([self.b2i[x] for x in token])
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)



if __name__ == "__main__":
    tokenizer = BPETokenizer()
    if False:
        vocab, merges = tokenizer.train(
            ROOT / "data/TinyStoriesV2-GPT4-train.txt",
            10000,
            ["<|endoftext|>"],
        )
        with open(ROOT / "submission/tinystories_vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        with open(ROOT / "submission/tinystories_merges.pkl", "wb") as f:
            pickle.dump(merges, f)
    else:
        vocab, merges = tokenizer.train(
            ROOT / "data/owt_train.txt",
            32000,
            ["<|endoftext|>"],
        )
        with open(ROOT / "submission/owt_vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        with open(ROOT / "submission/owt_merges.pkl", "wb") as f:
            pickle.dump(merges, f)