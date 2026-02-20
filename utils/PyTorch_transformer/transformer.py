import time
import logging

import torch
from torch import nn
from torch.nn import Linear, Dropout, Embedding

from encoder import Encoder
from decoder import Decoder
from positional_encoding import positional_encoding


logger_init = logging.getLogger(f"{__name__}.init")  # Logger for initialization
logger_forward = logging.getLogger(f"{__name__}.forward")  # Logger for forward pass

class Transformer(nn.Module):
    """
    Complete Transformer Model.
    
    A flexible implementation that supports:
        - Encoder-only models (like BERT)
        - Decoder-only models (like GPT)
        - Encoder-decoder models (like original Transformer for translation)
    
    The model can work with raw token IDs (creates embeddings internally) or
    pre-computed embeddings (useful for custom embedding strategies).
    
    Args:
        num_encoder_layers: Number of encoder layers (0 for decoder-only)
        num_decoder_layers: Number of decoder layers (0 for encoder-only)
        d_model: Model embedding dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        output_vocab_size: Size of output vocabulary
        seq_len: Maximum sequence length
        dropout_rate: Dropout probability
        input_vocab_size: Input vocabulary size (None if using pre-computed embeddings)
        target_vocab_size: Target vocabulary size (None if using pre-computed embeddings)
        use_positional_encoding: Whether to add positional encodings
        make_embadings: Legacy parameter (not used)
    """
    
    name = None
    iteration_id = 0

    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, d_model: int, num_heads: int, d_ff: int, 
                 output_vocab_size: int, seq_len: int, dropout_rate: float = 0.1, input_vocab_size: int | None = None, 
                 target_vocab_size: int | None = None, use_positional_encoding=True):
        super().__init__()

        self.d_model = d_model
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.output_vocab_size = output_vocab_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.use_positional_encoding = use_positional_encoding
        
        # Generate positional encodings
        pos_encoding = positional_encoding(d_model, seq_len)
        self.register_buffer("pos_encoding", pos_encoding)

        self.dropout = Dropout(dropout_rate)
        
        # Create embedding layers if vocabulary sizes are provided
        if input_vocab_size is not None:
            self.encoder_embedding = Embedding(input_vocab_size, d_model)

        if target_vocab_size is not None:
            self.decoder_embedding = Embedding(target_vocab_size, d_model)

        # Stack of encoder layers
        self.encoder = nn.ModuleList([Encoder(d_model, num_heads, d_ff, dropout_rate) 
                       for _ in range(num_encoder_layers)])

        # Stack of decoder layers
        self.decoder = nn.ModuleList([Decoder(d_model, num_heads, d_ff, dropout_rate) 
                       for _ in range(num_decoder_layers)])

        # Final linear layer to project to output vocabulary
        self.linear = Linear(d_model, output_vocab_size, bias=True)

        self._model_summary()

    def forward(self, args):
        """
        Forward pass through the complete Transformer.
        
        Args:
            args: Either a tuple (x_encoder, x_decoder) or a dictionary with keys:
                - 'encoder_input': Encoder input
                - 'decoder_input': Decoder input
                - 'attention_mask': Optional causal mask
                - 'encoder_padding_mask': Optional encoder padding mask
                - 'decoder_padding_mask': Optional decoder padding mask
            training: Whether in training mode
            
        Returns:
            Logits over output vocabulary of shape (batch_size, seq_len, output_vocab_size)
        """
        start_time = time.time()

        # Parse inputs (supports both dict and tuple formats)
        if isinstance(args, dict):
            x_encoder = args.get('encoder_input')
            x_decoder = args.get('decoder_input')
            attention_mask = args.get('attention_mask')
            encoder_padding_mask = args.get('encoder_padding_mask')
            decoder_padding_mask = args.get('decoder_padding_mask')
        else:
            x_encoder, x_decoder = args
            attention_mask = None
            encoder_padding_mask = None
            decoder_padding_mask = None

        # logger_forward.debug("InputData: x_encoder_shape=%s, x_decoder_shape=%s, attention_mask_shape=%s, got_encoder_padding_mask=%s, got_decoder_padding_mask=%s",
        #                      x_encoder.shape, x_decoder.shape, attention_mask.shape or None, encoder_padding_mask.shape if encoder_padding_mask is not None else None, decoder_padding_mask.shape if decoder_padding_mask is not None else None)

        if logger_forward.isEnabledFor(logging.DEBUG):
            logger_forward.debug("=" * 50)
            logger_forward.debug("FORWARD PASS START")
            logger_forward.debug("  Encoder input: %s", x_encoder.shape)
            logger_forward.debug("  Decoder input: %s", x_decoder.shape)
            
            # Маски — только если есть
            masks_info = []
            if attention_mask is not None:
                masks_info.append(f"attention({attention_mask.shape})")
            if encoder_padding_mask is not None:
                masks_info.append(f"enc_padding({encoder_padding_mask.shape})")
            if decoder_padding_mask is not None:
                masks_info.append(f"dec_padding({decoder_padding_mask.shape})")
            
            if masks_info:
                logger_forward.debug("  Masks: %s", ", ".join(masks_info))
            else:
                logger_forward.debug("  Masks: none")

        # ===== ENCODER PROCESSING =====
        if self.num_encoder_layers > 0:
            
            if logger_forward.isEnabledFor(logging.DEBUG):
                logger_forward.debug("Processing encoder (%d layers)...", self.num_encoder_layers)

            seq_len = x_encoder.size(1)

            # Check if we need to create embeddings or if they're pre-computed
            if self.input_vocab_size is None:
                # Pre-computed embeddings: expect shape (batch_size, seq_len, d_model)
                if x_encoder.ndim != 3:
                    logger_forward.critical(
                        "INPUT SHAPE ERROR: Expected pre-computed embeddings with shape "
                        "(batch_size, seq_len, d_model=%d), but got shape %s with %d dimensions. "
                        "Set input_vocab_size if you want to use token IDs instead.",
                        self.d_model, x_encoder.shape, x_encoder.ndim
                    )   
                    raise ValueError(
                        f"Invalid encoder input shape. Expected (batch, seq_len, {self.d_model}), "
                        f"got {x_encoder.shape}"
                    )
            else:
                # Token IDs: expect shape (batch_size, seq_len)
                if x_encoder.ndim != 2:
                    logger_forward.critical(
                        "INPUT SHAPE ERROR: Expected token IDs with shape (batch_size, seq_len), "
                        "but got shape %s with %d dimensions. vocab_size=%d is set, so embeddings "
                        "will be created automatically.",
                        x_encoder.shape, x_encoder.ndim, self.input_vocab_size
                    )
                    raise ValueError(
                        f"Invalid encoder input shape. Expected (batch, seq_len), got {x_encoder.shape}"
                    )
                x_encoder = self.encoder_embedding(x_encoder)

            # Scale embeddings by sqrt(d_model) (from original paper)
            x_encoder *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

            # Add positional encodings
            if self.use_positional_encoding:
                x_encoder += self.pos_encoding[:, :seq_len, :]
   
            # Apply dropout
            x_encoder = self.dropout(x_encoder)

            # Pass through all encoder layers
            for i, layer in enumerate(self.encoder):
                if logger_forward.isEnabledFor(logging.DEBUG):
                    logger_forward.debug("  Encoder layer %d/%d:", i+1, len(self.encoder))
                x_encoder = layer(x_encoder, padding_mask=encoder_padding_mask)

            # Logging encoder output
            if logger_forward.isEnabledFor(logging.DEBUG):
                logger_forward.debug("  Encoder output: %s", x_encoder.shape)
        
        # ===== DECODER PROCESSING =====
        if self.num_decoder_layers > 0:

            if logger_forward.isEnabledFor(logging.DEBUG):
                logger_forward.debug("Processing decoder (%d layers)...", self.num_decoder_layers)

            seq_len = x_decoder.size(1)

            # Check if we need to create embeddings or if they're pre-computed
            if self.target_vocab_size is None:
                # Pre-computed embeddings: expect shape (batch_size, seq_len, d_model)
                if x_decoder.ndim != 3:
                    logger_forward.critical(
                        "INPUT SHAPE ERROR: Expected pre-computed embeddings with shape "
                        "(batch_size, seq_len, d_model=%d), but got shape %s with %d dimensions. "
                        "Set input_vocab_size if you want to use token IDs instead.",
                        self.d_model, x_decoder.shape, x_decoder.ndim
                    )   
                    raise ValueError(
                        f"Invalid encoder input shape. Expected (batch, seq_len, {self.d_model}), "
                        f"got {x_decoder.shape}"
                    )
            else:
                # Token IDs: expect shape (batch_size, seq_len)
                if x_decoder.ndim != 2:
                    logger_forward.critical(
                        "INPUT SHAPE ERROR: Expected token IDs with shape (batch_size, seq_len), "
                        "but got shape %s with %d dimensions. vocab_size=%d is set, so embeddings "
                        "will be created automatically.",
                        x_decoder.shape, x_decoder.ndim, self.target_vocab_size
                    )
                    raise ValueError(
                        f"Invalid encoder input shape. Expected (batch, seq_len), got {x_decoder.shape}"
                    )
                x_decoder = self.decoder_embedding(x_decoder)

              
            # Scale embeddings by sqrt(d_model) (from original paper)
            x_decoder *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

            if self.use_positional_encoding:
                x_decoder += self.pos_encoding[:, :seq_len, :]

            # Apply dropout
            x_decoder = self.dropout(x_decoder)

            # Pass through all decoder layers
            for i, layer in enumerate(self.decoder):
                if logger_forward.isEnabledFor(logging.DEBUG):
                    logger_forward.debug("  Decoder layer %d/%d:", i+1, len(self.decoder))
                x_decoder = layer(x_decoder, x_encoder, 
                                 attention_mask=attention_mask, padding_mask=decoder_padding_mask)
            
            # Logging decoder output
            if logger_forward.isEnabledFor(logging.DEBUG):
                logger_forward.debug("  Decoder output: %s", x_decoder.shape)

        else:
            # If no decoder layers, use encoder output directly
            logger_forward.debug("No decoder layers defined, using encoder output directly")
            x_decoder = x_encoder

        # Final linear projection to output vocabulary
        output = self.linear(x_decoder)

        # Version for DEBUG level
        if logger_forward.isEnabledFor(logging.DEBUG):
            logger_forward.debug("  Final output: %s", output.shape)
            logger_forward.debug("FORWARD PASS COMPLETED")
            logger_forward.debug("=" * 50)
        elif logger_forward.isEnabledFor(logging.INFO):
            # Shorted version for INFO level
            logger_forward.info("Forward pass: %s -> %s (%s mode)",
                              x_encoder.shape, output.shape)

        if logger_forward.isEnabledFor(logging.DEBUG):
            elapsed = (time.time() - start_time) * 1000  # в миллисекундах
            logger_forward.debug("  Forward pass time: %.2f ms", elapsed)

        return output

    def _model_summary(self):
        logger_init.info("=" * 60)
        logger_init.info("TRANSFORMER MODEL SUMMARY")
        logger_init.info("=" * 60)
        logger_init.info("Architecture:")
        logger_init.info("  Encoder layers: %d", self.num_encoder_layers)
        logger_init.info("  Decoder layers: %d", self.num_decoder_layers)
        logger_init.info("  Total layers: %d", self.num_encoder_layers + self.num_decoder_layers)
        logger_init.info("")
        logger_init.info("Model dimensions:")
        logger_init.info("  d_model: %d", self.d_model)
        logger_init.info("  Input vocab size: %s", self.input_vocab_size or "pre-computed embeddings")
        logger_init.info("  Target vocab size: %s", self.target_vocab_size or "pre-computed embeddings")
        logger_init.info("")
        logger_init.info("Features:")
        logger_init.info("  Positional encoding: %s", "enabled" if self.use_positional_encoding else "disabled")
        
        logger_init.info("=" * 60)

    def __str__(self):
        return f"Name: {self.name}\nEncoding: \n\tnumber: {self.num_encoder_layers}\nDecoding: \n\tnumber:{self.num_decoder_layers}\nOutput_size: {self.output_vocab_size}"


if __name__ == '__main__':
    """
    Example usage: Training a simple encoder-only Transformer for sequence classification.
    
    This example demonstrates:
        - Creating a Transformer with 2 encoder layers and no decoder
        - Training on dummy data with pre-computed embeddings
        - Using the model for sequence-level predictions
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    # from logger.utils.logger import setup_logging

    # setup_logging()

    logging.getLogger(f'{__name__}.init').setLevel(logging.INFO)
    logging.getLogger(f'{__name__}.forward').setLevel(logging.WARNING) 
    logging.getLogger(f'{__name__}.attention').setLevel(logging.WARNING)

    # ===== MODEL CONFIGURATION =====
    num_encoders = 2  # Number of encoder layers
    num_decoders = 2  # Number of decoder layers
    d_model = 16  # Embedding dimension
    num_heads = 2  # Number of attention heads
    d_ff = 64  # Feed-forward hidden dimension
    output_vocab_size = 5  # Binary classification
    seq_len = 10  # Sequence length

    # ===== TRAINING CONFIGURATION =====
    BATCH_SIZE = 32
    EPOCHS = 50

    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Example causal/look-ahead mask (for autoregressive tasks)
    # Each row shows which positions a token can attend to
    attention_mask = torch.tril(torch.ones(seq_len, seq_len))
    attention_mask = attention_mask.to(device)

    # ===== MODEL CREATION =====
    transformer = Transformer(
        num_encoders,
        num_decoders, 
        d_model, 
        num_heads, 
        d_ff, 
        output_vocab_size, 
        seq_len
    )
    transformer.name = "Test_model"
    transformer.to(device)

    criterion = nn.CrossEntropyLoss()  # например, для NLP
    optimizer = optim.Adam(transformer.parameters(), lr=1e-4)

    # ===== DUMMY DATA GENERATION =====
    # Create random input data (pre-computed embeddings)
    x_train = torch.randn((175, seq_len, d_model)).to(device)
    x_test = torch.randn((59, seq_len, d_model)).to(device)

    # Create labels
    y_train = torch.randint(0, output_vocab_size, (175, seq_len)).to(device)
    y_test = torch.randint(0, output_vocab_size, (59, seq_len)).to(device)

    train_padding_mask = (torch.rand(175, seq_len) > 0.2).int().to(device)
    test_padding_mask = (torch.rand(59, seq_len) > 0.2).int().to(device)
    
    # ===== PREPARE TRAINING DATA =====
    # Model expects dictionary input with encoder_input and decoder_input

    train_dataset  = TensorDataset(
        x_train, # encoder input
        x_train, # dacoder input
        y_train, # 
        train_padding_mask, # encoder pad mask
        train_padding_mask  # decoder pad mask
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ===== PREPARE VALIDATION DATA =====
    val_dataset  = TensorDataset(
        x_test, # encoder input
        x_test, # dacoder input
        y_test,
        test_padding_mask, # encoder pad mask
        test_padding_mask  # decoder pad 
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ===== TRAIN MODEL =====
    for epoch in range(EPOCHS):
        transformer.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        for enc_input, dec_input, label,  enc_pad_mask, dec_pad_mask in train_loader:
            input = {
                "encoder_input": enc_input,
                "decoder_input": dec_input,
                "attention_mask": attention_mask,
                "encoder_padding_mask": enc_pad_mask,
                "decoder_padding_mask": dec_pad_mask
            }
            
            optimizer.zero_grad()
            output = transformer(input)

            loss = criterion(
                output.reshape(-1, output.size(-1)),
                label.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * label.size(0)
            
            preds = output.argmax(dim=-1)
            correct += (preds == label).sum().item()
            total += label.numel()

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        train_aссuracy = correct / total
        train_accuracies.append(train_aссuracy)


        # Validation 
        transformer.eval()
        running_loss = 0.0
        with torch.no_grad():
            for enc_input, dec_input, label,  enc_pad_mask, dec_pad_mask in val_loader:
                input = {
                    "encoder_input": enc_input,
                    "decoder_input": dec_input,
                    "attention_mask": attention_mask,
                    "encoder_padding_mask": enc_pad_mask,
                    "decoder_padding_mask": dec_pad_mask
                }
                
                output = transformer(input)

                loss = criterion(
                    output.reshape(-1, output.size(-1)),
                    label.reshape(-1)
                )
                running_loss += loss.item() * label.size(0)

                preds = output.argmax(dim=-1)
                correct += (preds == label).sum().item()
                total += label.numel()

        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        val_aссuracy = correct / total
        val_accuracies.append(val_aссuracy)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f}, acc: {train_aссuracy:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_aссuracy:.4f}"
        )