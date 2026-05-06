import torch
import numpy as np
import logging

from .utils.vocabulary import load_vocabulary # temp
from .utils.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

class MyModel:
    def __init__(self, seq_len=512):
        logger.info("MyModel init")
        try:
            self.model = None # DecoderOnly
            self.id_token_vocabulary, self.token_id_vocabulary = load_vocabulary('chat/data/vocabulary/bpe_vocabulary.json') # some path to file
            logger.debug(f"Vocabulary loaded | size={len(self.id_token_vocabulary)}")

            self._tokenizer = Tokenizer()
            logger.debug("Tokenizer initialized")

            self.seq_len = seq_len
            logger.info(f"MyModel initialized | seq_len={seq_len}")
        except Exception as e:
            logger.error(f"MyModel init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    # Input processing
    def _tokenize(self, text: str) -> list:
        # Gets only 1 input type from each context list
        tokens = self._tokenizer.tokenize(text)
        return [self._token_to_ID(token) for token in tokens]

    def _token_to_ID(self, token: str) -> int:
        # Finds id of token in vocabulary
        return self.token_id_vocabulary.get(token, 0)

    def _preprocess(self, context_list: list[dict]) -> list:
        # context_list[-1] = {
        #     '/hint': '',
        #     '/sys': '',
        #     '/cv': '',
        #     '/user': '',
        #     '/model': ''
        # }
        logger.debug(f"Pre-processing context | context_len={len(context_list)}")
        try:
            tokens = []

            for context in context_list:
                for key, value in context.items():
                    if value:
                        tokens.append(self._token_to_ID(key))
                        tokens.extend(self._tokenize(value))
            tokens.append(1)

            result = tokens[-self.seq_len:]
            logger.debug(f"Tokens created | count={len(result)}")
            return result
        except Exception as e:
            logger.error(f"Pre-process error: {type(e).__name__}: {e}", exc_info=True)
            raise
    
    def _fill(self, tokens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:

        padding_mask = self._make_padding(tokens)

        for _ in range(self.seq_len - len(tokens)):
            tokens.append(0)

        return torch.tensor([tokens]), padding_mask # size(1, seq_len), size(1, seq_len) 

    def _make_padding(self, tokens: list[int]) -> torch.Tensor:
        return torch.concat([
            torch.zeros((1, len(tokens))), 
            torch.ones((1, self.seq_len - len(tokens)))], 
            
            dim=1).bool()
    
    # Output processing
    def _postprocess(self, ids: list[int]) -> str:
        tokens = self._decrypt(ids)

        return self._concat(tokens)

    def _decrypt(self, ids: list[int]) -> list[str]:
        return [self._ID_to_token(id) for id in ids]

    def _ID_to_token(self, id: int) -> str:
        return self.id_token_vocabulary.get(id, '')

    def _concat(self, tokens:list[str]) -> str:
        text = ''

        for token in tokens:
            text += token.replace('</w>', ' ')

        return text
            
    # 
    def _generate(self, tokens: list) -> list:
        # Луп который будет заниматься генерацией
        # Он самостоятельно будет генерироваать padding для каждой итерации
        # Вернет список нагенерированных токенов
        last_token = None
        
        generated_seq = []

        while last_token != 1:
            seq, pad = self._fill(tokens)

            y = self.model(seq, pad)
            pred = self._pred_processing(y, len(tokens))

            last_token = pred
            tokens.append(pred)
            generated_seq.append(pred)

        return generated_seq


    def _pred_processing(self, y: np.ndarray, req_len: int) -> int:
        last_token_posib = y[req_len-1]
        return np.argmax(last_token_posib)

    def __call__(self, x: list[dict]) -> str:
        # Занимается предобработткой текста для последующуй генерации
        # Конвертирует сгенерированные токены обратно в текст

        tokens = self._preprocess(x)

        if self.model: # While model is not done
            y = self._generate(tokens)
        else:
            y = tokens

        return self._postprocess(y)

if __name__ == '__main__':
    text  =  'Hello, my dog is cute'

    chat_module = ChatModule(20)

    prep = chat_module._preprocess([{
        '/user': text
    }])
    tokens, pad = chat_module._fill(prep)

    print('Tokens:', tokens.tolist(),'\n' \
          'Padding:', pad.tolist())
    print('Tokens_shape:', tokens.shape, '\n' \
          'Padding_shape', pad.shape)

    decr = chat_module._postprocess(tokens.squeeze().tolist())

    print('Output:', decr)
