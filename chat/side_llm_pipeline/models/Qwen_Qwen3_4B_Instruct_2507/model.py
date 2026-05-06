import os

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_TOKEN"] = "REMOVED_SECRET"


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
        print("Initializing Model...")
        self.model_name = "Qwen/Qwen3-4B-Instruct-2507"
        self.device = "cuda"  # for GPU usage or "cpu" for CPU usage

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir="chat/side_llm_pipeline/models/Qwen_Qwen3_4B_Instruct_2507")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir="chat/side_llm_pipeline/models/Qwen_Qwen3_4B_Instruct_2507",
            torch_dtype="auto",
            device_map="auto"
        )
        print("Model initialized.")

    def __call__(self, prompt: str):
        print("Processing prompt")
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

        print("Model generating...")
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=256)
        print("Model generated IDs")

        # Get and decode the output
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        decode_message = self.tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(decode_message)
        resp = decode_message.split('</think>')
        if len(resp) > 1:
            return resp[1]
        return resp[0]