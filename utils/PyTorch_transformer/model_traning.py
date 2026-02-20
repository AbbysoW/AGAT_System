import pandas as pd
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from transformer import Transformer
from stream_dataset import StreamDataset


def get_time(start, now) -> str:
    time = now - start

    if time * 1000 > 1000:
        time = int(time)
        return f"{time} s"
    else:
        time = int(time * 1000)
        return f"{time} ms"

# ===== MODEL CONFIGURATION =====
NUM_ENCODERS = 0
NUM_DECODERS = 2
D_MODEL = 128
NUM_HEADS = 8
D_FF = 256
OUTPUT_VACABULARY_SIZE = 393528
TARGET_VACABULARY_SIZE = 393528
SEQ_LEN = 128

# NUM_ENCODERS = 0
# NUM_DECODERS = 2
# D_MODEL = 512
# NUM_HEADS = 8
# D_FF = 256
# OUTPUT_VACABULARY_SIZE = 393528
# TARGET_VACABULARY_SIZE = 393528
# SEQ_LEN = 512

BATCH_SIZE = 2
EPOCHS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Example causal/look-ahead mask (for autoregressive tasks)
# Each row shows which positions a token can attend to
attention_mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN))
attention_mask = attention_mask.to(device)

# ===== MODEL CREATION =====
transformer = Transformer(
    num_encoder_layers=NUM_ENCODERS,
    num_decoder_layers=NUM_DECODERS,
    d_model=D_MODEL,
    d_ff=D_FF,
    num_heads=NUM_HEADS,
    output_vocab_size=OUTPUT_VACABULARY_SIZE,
    target_vocab_size=TARGET_VACABULARY_SIZE,
    seq_len=SEQ_LEN
)
transformer.to(device)

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(transformer.parameters(), lr=1e-4)

# ===== PREPARE TRAINING DATA =====
# Model expects dictionary input with encoder_input and decoder_input

npy_path = 'E:/Documents/VS/Python/AGAT/AGAT_System/Traning data/DataSet.npy'

data = np.load(npy_path, mmap_mode="r")

train_dataset = StreamDataset(
    data,
    SEQ_LEN,
    attention_mask,
    SEQ_LEN,
    split=0.75,
    device=device
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE
)

val_dataset = StreamDataset(
    data,
    SEQ_LEN,
    attention_mask,
    SEQ_LEN,
    split=0.75,
    is_val=True,
    device=device
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

# ===== TRAIN MODEL =====
train_accuracies, val_accuracies = [], []
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    transformer.train()

    start_time = time.time()

    running_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0

    # Training phase
    for input, label in train_loader:
        
        optimizer.zero_grad()
        output = transformer(input)

        loss = criterion(
            output.reshape(-1, output.size(-1)),
            label.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * label.size(0)
        num_batches += 1
        
        preds = output.argmax(dim=-1)
        correct += (preds == label).sum().item()
        total += label.numel()

        train_loss = running_loss / num_batches
        train_aссuracy = correct / total

        batch_time = get_time(start_time, time.time())

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"time: {batch_time} | "
            f"Train loss: {train_loss:.4f}, acc: {train_aссuracy:.4f}",
            end="/r"
        )

    train_losses.append(train_loss)
    train_accuracies.append(train_aссuracy)


    # Validation 
    transformer.eval()
    running_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for input, label in val_loader:
            
            output = transformer(input)

            loss = criterion(
                output.reshape(-1, output.size(-1)),
                label.reshape(-1)
            )
            running_loss += loss.item() * label.size(0)
            num_batches += 1

            preds = output.argmax(dim=-1)
            correct += (preds == label).sum().item()
            total += label.numel()

            val_loss = running_loss / num_batches
            val_aссuracy = correct / total

            batch_time = get_time(start_time, time.time())

            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"time: {batch_time} | "
                f"Train loss: {train_loss:.4f}, acc: {train_aссuracy:.4f} | "
                f"Val loss: {val_loss:.4f}, acc: {val_aссuracy:.4f}",
                end="/r"
            )

    val_losses.append(val_loss)
    val_accuracies.append(val_aссuracy)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"time: {batch_time} | "
        f"Train loss: {train_loss:.4f}, acc: {train_aссuracy:.4f} | "
        f"Val loss: {val_loss:.4f}, acc: {val_aссuracy:.4f}"
    )

