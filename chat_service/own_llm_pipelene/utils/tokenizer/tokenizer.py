import re
import logging

from pathlib import Path

logger = logging.getLogger(__name__)

class BPE:
    instruction = []

    _WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+")


    def __init__(self):
        logger.debug("BPE init")
        try:
            self.load_instruction(Path('chat/data/vocabulary/bpe_merges.txt'))
            logger.debug(f"BPE instructions loaded | count={len(self.instruction)}")
        except Exception as e:
            logger.error(f"BPE init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def load_instruction(self, path: Path):
        logger.debug(f"Loading BPE instructions | path={path}")
        try:
            with open(path, 'r') as f:
                self.instruction = [tuple(line.split()) for line in f]
            logger.debug(f"Instructions loaded | count={len(self.instruction)}")
        except Exception as e:
            logger.error(f"Load instructions error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def tokenize(self, text) -> list:
        logger.debug(f"Tokenizing | text_len={len(text)}")
        try:
            tokens = self._pretokenize(text)
            for step in self.instruction:
                i = 0
                while i < len(tokens) - 1:
                    if (tokens[i], tokens[i + 1]) == step:
                        tokens[i] = tokens[i] + tokens[i + 1]
                        del tokens[i + 1]
                    else:
                        i += 1

            logger.debug(f"Tokenization complete | token_count={len(tokens)}")
            return tokens
        except Exception as e:
            logger.error(f"Tokenize error: {type(e).__name__}: {e}", exc_info=True)
            raise
    
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