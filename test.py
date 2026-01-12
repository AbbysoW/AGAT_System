import torch

# Исходные данные
batch_size = 2
seq_len = 4
embedding_dim = 8

# Входные тензоры
inputs = torch.randn(batch_size, seq_len, embedding_dim)  # (batch, seq, embedding)
padding_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])  # (batch, seq)
attention_mask = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])  # (seq, seq)

# Расширение Padding Mask до формы (batch, seq, seq)
padding_mask_expanded = padding_mask.unsqueeze(2).expand(-1, -1, seq_len)  # (batch, seq, seq)

# Объединение Attention Mask и Padding Mask
# Используем побитовую операцию, чтобы оставить только элементы, которые не являются padding
combined_mask = padding_mask_expanded * attention_mask  # (batch, seq, seq)

# Теперь combined_mask можно использовать как mask для self-attention
