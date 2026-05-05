import re

from pathlib import Path

class BPE:
    instruction = []

    _WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+")


    def __init__(self):
        self.load_instruction(Path('chat/data/vocabulary/bpe_merges.txt'))

    def load_instruction(self, path: Path):
        with open(path, 'r') as f:
            self.instruction = [tuple(line.split()) for line in f]

    def tokenize(self, text) -> list:
        tokens = self._pretokenize(text)
        for step in self.instruction:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == step:
                    tokens[i] = tokens[i] + tokens[i + 1]
                    del tokens[i + 1]
                else:
                    i += 1

        return tokens
    
    def _pretokenize(self, text: str) -> list:
        result = []
        for w in self._WORD_RE.findall(text):
            result.extend(list(w) + ["</w>"])
        return result


# WRAP
class Tokenizer:
    tokenizer = BPE()

    def load_instruction(self, path: Path):
        return self.tokenizer.load_instruction(path)

    def tokenize(self, text) -> list:
        return self.tokenizer.tokenize(text)
    
if __name__ == '__main__':
    
    text  =  'Hello, my dog is cute'

    tokenizer = Tokenizer()

    print(tokenizer.tokenize(text))