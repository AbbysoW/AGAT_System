import transformers

class Model:
    def __init__(self):
        self.sampling_rate = 16000
        self.turns = [
        {
            "role": "system",
            "content": "You are a friendly and helpful character. You love to answer questions for people."
        },
        ]

        self.pipeline = transformers.pipeline(model='fixie-ai/ultravox-v0_5-llama-3_2-1b', trust_remote_code=True)

    def __call__(self, audio):
        return self.pipeline({'audio': audio, 'turns': self.turns, 'sampling_rate': self.sampling_rate}, max_new_tokens=30)