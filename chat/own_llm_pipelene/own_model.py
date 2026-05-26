import torch
import numpy as np
import logging

from .utils.vocabulary import load_vocabulary # temp
from .utils.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

class MyModel:
    def __init__(self, seq_len=512):
        logger.info(f"Initializing MyModel | seq_len={seq_len}")
        try:
            logger.debug("Loading model components...")
            
            logger.debug("Initializing model decoder...")
            self.model = None # DecoderOnly
            logger.debug("✓ Model decoder placeholder set")
            
            logger.debug("Loading vocabulary...")
            self.id_token_vocabulary, self.token_id_vocabulary = load_vocabulary('chat/data/vocabulary/bpe_vocabulary.json') # some path to file
            logger.info(f"✓ Vocabulary loaded | vocab_size={len(self.id_token_vocabulary)}")
            logger.debug(f"Vocabulary keys sample: {list(self.id_token_vocabulary.items())[:3]}")

            logger.debug("Initializing tokenizer...")
            self._tokenizer = Tokenizer()
            logger.debug("✓ Tokenizer initialized")

            self.seq_len = seq_len
            logger.info(f"✓ MyModel initialized successfully | seq_len={seq_len}")
        except Exception as e:
            logger.error(f"✗ MyModel init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    # Input processing
    def _tokenize(self, text: str) -> list:
        """Convert text to tokens using tokenizer."""
        logger.debug(f"Tokenizing text | len={len(text)}")
        # Gets only 1 input type from each context list
        tokens = self._tokenizer.tokenize(text)
        logger.debug(f"Tokenization complete | token_count={len(tokens)}")
        return [self._token_to_ID(token) for token in tokens]

    def _token_to_ID(self, token: str) -> int:
        """Convert token string to ID using vocabulary."""
        token_id = self.token_id_vocabulary.get(token, 0)
        logger.debug(f"Token to ID | token='{token}' | id={token_id}")
        # Finds id of token in vocabulary
        return token_id

    def _preprocess(self, context_list: list[dict]) -> list:
        """Preprocess context into tokens."""
        logger.debug(f"Pre-processing context | context_len={len(context_list)}")
        # context_list[-1] = {
        #     '/hint': '',
        #     '/sys': '',
        #     '/cv': '',
        #     '/user': '',
        #     '/model': ''
        # }
        try:
            tokens = []

            for idx, context in enumerate(context_list):
                logger.debug(f"Processing context {idx+1}/{len(context_list)} | keys={list(context.keys())}")
                for key, value in context.items():
                    if value:
                        logger.debug(f"Processing key: {key} | value_len={len(str(value))}")
                        tokens.append(self._token_to_ID(key))
                        tokens.extend(self._tokenize(value))
            tokens.append(1)

            result = tokens[-self.seq_len:]
            logger.info(f"✓ Tokens created | total={len(tokens)} | seq_len={len(result)}")
            return result
        except Exception as e:
            logger.error(f"✗ Pre-process error: {type(e).__name__}: {e}", exc_info=True)
            raise
    
    def _fill(self, tokens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill tokens to sequence length and create padding mask."""
        logger.debug(f"Filling tokens to seq_len | current_len={len(tokens)} | target_len={self.seq_len}")

        padding_mask = self._make_padding(tokens)

        for _ in range(self.seq_len - len(tokens)):
            tokens.append(0)

        logger.debug(f"Padding applied | padded_len={len(tokens)}")
        return torch.tensor([tokens]), padding_mask # size(1, seq_len), size(1, seq_len) 

    def _make_padding(self, tokens: list[int]) -> torch.Tensor:
        """Create padding mask for tokens."""
        logger.debug(f"Creating padding mask | token_len={len(tokens)} | seq_len={self.seq_len}")
        mask = torch.concat([
            torch.zeros((1, len(tokens))), 
            torch.ones((1, self.seq_len - len(tokens)))], 
            
            dim=1).bool()
        logger.debug(f"Padding mask created | shape={mask.shape}")
        return mask
    
    # Output processing
    def _postprocess(self, ids: list[int]) -> str:
        """Postprocess token IDs into text."""
        logger.debug(f"Post-processing output | token_count={len(ids)}")
        tokens = self._decrypt(ids)
        result = self._concat(tokens)
        logger.info(f"✓ Post-processing complete | result_len={len(result)}")
        return result

    def _decrypt(self, ids: list[int]) -> list[str]:
        """Convert token IDs to token strings."""
        logger.debug(f"Decrypting {len(ids)} token IDs")
        return [self._ID_to_token(id) for id in ids]

    def _ID_to_token(self, id: int) -> str:
        """Convert token ID to token string."""
        token = self.id_token_vocabulary.get(id, '')
        logger.debug(f"ID to token | id={id} | token='{token}'")
        return token

    def _concat(self, tokens:list[str]) -> str:
        """Concatenate tokens into final text."""
        logger.debug(f"Concatenating {len(tokens)} tokens")
        text = ''

        for token in tokens:
            text += token.replace('</w>', ' ')

        logger.debug(f"Text concatenation complete | final_len={len(text)}")
        return text
            
    # 
    def _generate(self, tokens: list) -> list:
        """Generate tokens until end-of-sequence token."""
        logger.info("🔄 Starting token generation")
        logger.debug(f"Initial tokens: {len(tokens)} | seq_len: {self.seq_len}")
        
        # Луп который будет заниматься генерацией
        # Он самостоятельно будет генерироваать padding для каждой итерации
        # Вернет список нагенерированных токенов
        last_token = None
        
        generated_seq = []
        iteration = 0

        while last_token != 1:
            iteration += 1
            logger.debug(f"Generation iteration #{iteration} | generated_tokens={len(generated_seq)}")
            
            seq, pad = self._fill(tokens)
            logger.debug(f"Sequence filled | shape={seq.shape} | padding_shape={pad.shape}")

            y = self.model(seq, pad)
            logger.debug(f"Model output received | shape={y.shape}")
            
            pred = self._pred_processing(y, len(tokens))
            logger.debug(f"Prediction: {pred}")

            last_token = pred
            tokens.append(pred)
            generated_seq.append(pred)
            
            if iteration > 1000:
                logger.warning("⚠ Generation stopped - exceeded 1000 iterations")
                break

        logger.info(f"✓ Token generation complete | total_generated={len(generated_seq)}")
        return generated_seq


    def _pred_processing(self, y: np.ndarray, req_len: int) -> int:
        """Process model output to select next token."""
        logger.debug(f"Processing predictions | output_shape={y.shape} | req_len={req_len}")
        last_token_posib = y[req_len-1]
        prediction = np.argmax(last_token_posib)
        logger.debug(f"Argmax prediction: {prediction}")
        return prediction

    def __call__(self, x: list[dict]) -> str:
        """Main inference pipeline."""
        logger.info(f"📤 MyModel inference | context_len={len(x)}")
        logger.debug(f"Context: {x}")
        
        try:
            # Занимается предобработткой текста для последующуй генерации
            # Конвертирует сгенерированные токены обратно в текст

            logger.debug("Starting pre-processing...")
            tokens = self._preprocess(x)
            logger.info(f"✓ Pre-processing complete | tokens={len(tokens)}")

            if self.model: # While model is not done
                logger.info("Model available - starting generation...")
                y = self._generate(tokens)
            else:
                logger.warning("⚠ Model not loaded - using tokens directly")
                y = tokens

            logger.debug("Starting post-processing...")
            result = self._postprocess(y)
            logger.info(f"✓ Inference complete | result_len={len(result)}")
            
            return result
        except Exception as e:
            logger.error(f"✗ Inference error: {type(e).__name__}: {e}", exc_info=True)
            raise

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
