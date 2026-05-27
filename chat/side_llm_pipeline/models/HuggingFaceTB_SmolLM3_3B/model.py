import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

logger = logging.getLogger(__name__)


class Model:

    SYSTEM_PROMPT = """
    Ты ассистент по имени Агат. Обращайся к себе в женском роде.

    СТРОГИЕ ПРАВИЛА:
    - Не выводи служебные токены: /sys, /cv, /hint, /user, /model
    - Отвечай только финальным ответом
    - Языки общения только английский и русский.

    ФОРМАТ ОТВЕТА:
    Короткий текст без префиксов и тегов.

    Тон: допускается лёгкая язвительность.

    
    Контекст ниже — это формат входных данных, НЕ формат ответа:

    /sys — Нынешнее состояние системы пользователя(может отсутствовать),
    /cv — Информация из компьютерного зрения (может отсутствовать), 
    /hint — подсказки, используй их если есть (может отсутствовать), 
    /user — сообщение пользователя. Содержит наименование пользователя. Если видишь 'Владелец', обращайся к нему как Андрей.
    /model — предыдущий ответ модели. 
    """

    def __init__(self):
        logger.info("Loading SmolLM3 model")
        try:
            self.model_name = "HuggingFaceTB/SmolLM3-3B"
            self.device = "cpu"

            logger.debug(f"Loading tokenizer | model={self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=Path(__file__).parent  / "HuggingFaceTB_SmolLM3_3B")
            logger.debug("Tokenizer loaded")
            
            logger.debug(f"Loading model | model={self.model_name} device={self.device}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=Path(__file__).parent  / "HuggingFaceTB_SmolLM3_3B"
            ).to(self.device)
            logger.info("SmolLM3 model loaded successfully")
        except Exception as e:
            logger.error(f"SmolLM3 load error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def __call__(self, prompt: str):
        logger.debug(f"SmolLM3 inference | prompt_len={len(prompt)}")
        try:
            messages_think = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

            text = self.tokenizer.apply_chat_template(
                messages_think,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            logger.debug("Tokenization complete")

            generated_ids = self.model.generate(**model_inputs, max_new_tokens=1024)
            logger.debug("Generation complete")

            # Get and decode the output
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
            decode_message = self.tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            logger.info(f"SmolLM3 result | result_len={len(decode_message)}")
            
            resp = decode_message.split('<tool_call>')
            if len(resp) > 1:
                return resp[1]
            return resp[0]
        except Exception as e:
            logger.error(f"SmolLM3 inference error: {type(e).__name__}: {e}", exc_info=True)
            raise