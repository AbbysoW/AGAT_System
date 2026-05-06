from .side_llm_pipeline import SideModel

class ChatModule:
    def __init__(self):
        self.model = SideModel()

    def __call__(self, x: list[dict]) -> str:
        return self.model(x)    
    
if __name__ == '__main__':

    context = [
        {
            '/hint': None,
            '/sys': None,
            '/cv': None,
            '/user': "Владелец: Привет Агат! Тестовый запуск. Посчитай от 1 до 5",
            '/model': None
        }
    ]

    chat = ChatModule()
    print(chat(context))