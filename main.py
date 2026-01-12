import chat
import rag
# from stt import 

from data_processing import preprocess, seq_to_ids, ids_to_seq, make_padding_mask
from vocabulary import vocabulary

# Из STT получить 
request = ''
request_tokens = preprocess()
request_tokens = seq_to_ids(request_tokens, vocabulary)

# Из RAG получить TEXT
info = ''
info_tokens = preprocess()
info_tokens = seq_to_ids(info_tokens, vocabulary)


context = ''
context_tokens = preprocess(context)
context_tokens = seq_to_ids(context_tokens, vocabulary)

seq = context_tokens + info_tokens + request_tokens

padding  = make_padding_mask(seq, chat.model.seq_len)

y = chat.model()