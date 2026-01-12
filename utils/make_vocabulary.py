import json
import pickle as pkl

from collections  import Counter

import sys
from pathlib import Path

# Добавляем родительскую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing import preprocess, count_tokens, build_vocab

TEXT_PATH = 'E:\Documents\VS\Python\AGAT\AGAT_System\Traning data/temp.txt'

# Вариант 1: JSON (универсальный)
VOCABULARY = 'E:\Documents\VS\Python\AGAT\AGAT_System/vocabulary/'



print("Начинаем обработку файла...")

CHARS = [
    # Русский алфавит (lower)
    'а','б','в','г','д','е','ё','ж','з','и','й','к','л','м','н','о','п',
    'р','с','т','у','ф','х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я',

    # Русский алфавит (UPPER)
    'А','Б','В','Г','Д','Е','Ё','Ж','З','И','Й','К','Л','М','Н','О','П',
    'Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ы','Ь','Э','Ю','Я',

    # Английский алфавит (lower)
    'a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z',

    # Английский алфавит (UPPER)
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',

    # Цифры
    '0','1','2','3','4','5','6','7','8','9',

    # Знаки препинания и спецсимволы
    '!','"','#','$','%','&',"'",'(',')','*','+',
    ',', '-', '.', '/', ':',';','<','=','>','?',
    '@','[','\\',']','^','_','`','{','|','}','~',

    # Русские и типографские символы
    '«','»','—','…'
]

# Потоковая обработка с минимальным потреблением памяти
with open(TEXT_PATH, 'r', encoding='utf-8') as f:
    token_counts = Counter()
    for i, line in enumerate(f):
        tokens = preprocess(line)
        token_counts = count_tokens(tokens, token_counts)
        print(f'line: {i}', end='\r')
    print("")

vocabulary = build_vocab(token_counts, min_freq= 20, white_list= CHARS)

# JSON - для совместимости
with open(VOCABULARY + "vocab20.json", 'w', encoding='utf-8') as f:
    json.dump(vocabulary, f, ensure_ascii=False)

# Pickle - для скорости загрузки в Python
with open(VOCABULARY + "vocab20.pkl", 'wb') as f:
    pkl.dump(vocabulary, f)


vocabulary = build_vocab(token_counts, min_freq= 25, white_list= CHARS)

# JSON - для совместимости
with open(VOCABULARY + "vocab25.json", 'w', encoding='utf-8') as f:
    json.dump(vocabulary, f, ensure_ascii=False)

# Pickle - для скорости загрузки в Python
with open(VOCABULARY + "vocab25.pkl", 'wb') as f:
    pkl.dump(vocabulary, f)