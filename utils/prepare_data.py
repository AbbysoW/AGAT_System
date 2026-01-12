import numpy as np
from typing import List, Dict
import re

from data_processing import preprocess, seq_to_ids

import pickle as pkl

from datetime import datetime



TRANING_PATH = 'E:\Documents\VS\Python\AGAT\AGAT_System\Traning data/'
VOCABULARY = 'E:\Documents\VS\Python\AGAT\AGAT_System/vocabulary/vocab.pkl'

with open(VOCABULARY, 'rb') as f:
    vocabulary = pkl.load(f)

def process_dataset_optimized(file_path: str, vocabulary: Dict, batch_size: int = 10000):
    """
    Оптимизированная обработка датасета с батчингом
    """
    # Предобработка словаря
    if vocabulary.get(0):  # Если словарь id->token
        vocab_token_to_id = {v: k for k, v in vocabulary.items()}
    else:
        vocab_token_to_id = vocabulary
    
    unk_id = vocab_token_to_id.get('<UNK>', 1)
    
    # Батчевая обработка
    dataset_chunks = []
    batch_tokens = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Обработка строки
            tokens = preprocess(line)
            token_ids = seq_to_ids(tokens, vocab_token_to_id)
            
            batch_tokens.extend(token_ids)
            
            # Сохраняем батч когда накопилось достаточно
            if len(batch_tokens) >= batch_size:
                dataset_chunks.append(np.array(batch_tokens, dtype=np.int32))
                batch_tokens = []
            
            if i % 1000 == 0:
                print(f"Processed {i} lines", end='\r')
    
    # Добавляем остаток
    if batch_tokens:
        dataset_chunks.append(np.array(batch_tokens, dtype=np.int32))
    
    # Объединяем все батчи
    dataset = np.concatenate(dataset_chunks)
    
    return dataset

# Использование:
dataset = process_dataset_optimized(
    TRANING_PATH + 'row_data.txt', 
    vocabulary,
    batch_size=50000
)

path = TRANING_PATH + 'DataSet_' + datetime.now().strftime("%d-%m-%Y") + '.npy'

with open(path, 'wb') as f:
    np.save(f, dataset)