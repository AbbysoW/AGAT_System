import time
import logging

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, LayerNormalization, Embedding
from keras.models import Model

import numpy as np


logger_init = logging.getLogger(f"{__name__}.init")  # Logger for initialization
logger_forward = logging.getLogger(f"{__name__}.forward")  # Logger for forward pass
logger_attention = logging.getLogger(f"{__name__}.attention")  # Logger for MultiHeadAttention


def positional_encoding(d_model: int, seq_len: int):
    """
    Generates sinusoidal positional encodings for the Transformer model.
    
    This function creates position-dependent patterns that allow the model to understand
    the order of tokens in a sequence, since attention mechanism itself is permutation-invariant.
    
    Args:
        d_model: Embedding dimension of the model
        seq_len: Maximum sequence length
        
    Returns:
        Tensor of shape (1, seq_len, d_model) containing positional encodings
        
    Mathematical formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    # Create position indices [0, 1, 2, ..., seq_len-1]
    angle_rads = np.arange(seq_len)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))

    # Apply sin to even indices in the array
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Apply cos to odd indices in the array
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    # Add batch dimension
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(keras.layers.Layer):
    """
    Multi-Head Attention mechanism - the core component of the Transformer architecture.
    
    This layer splits the input into multiple attention heads, allowing the model to jointly
    attend to information from different representation subspaces at different positions.
    
    Args:
        d_model: Total dimension of the model (must be divisible by num_heads)
        num_heads: Number of attention heads
    """
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0

        # Dimension of each attention head
        self.depth = d_model // num_heads

        # Linear projections for Query, Key, Value
        self.wq = Dense(d_model)  # Query projection
        self.wk = Dense(d_model)  # Key projection
        self.wv = Dense(d_model)  # Value projection
        
        # Final linear projection after concatenating all heads
        self.dense = Dense(d_model)

        logger_init.info("MultiHeadAttention initialized")
        logger_init.info("MultiHeadAttentionlConfig: d_model=%s, heads=%s", 
                         d_model, num_heads)

    def split_heads(self, x, batch_size: int):
        """
        Splits the last dimension into (num_heads, depth) and transposes for parallel processing.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            batch_size: Size of the batch
            
        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, depth)
        """
        # Reshape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        # Transpose: (batch_size, seq_len, num_heads, depth) -> (batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def attention_function(self, q, k, v, mask=None):
        """
        Computes scaled dot-product attention.
        
        This calculates how much each word in a sequence should "attend to" every other word.
        The attention weights determine which parts of the input are most relevant.
        
        Args:
            q: Query matrix of shape (..., seq_len_q, depth)
            k: Key matrix of shape (..., seq_len_k, depth)
            v: Value matrix of shape (..., seq_len_v, depth)
            mask: Optional mask to prevent attention to certain positions
            
        Returns:
            output: Attention-weighted values
            attention_weights: Attention probability distribution
            
        Formula: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
        """
        # Calculate attention scores: Q * K^T
        matrix_mult = tf.matmul(q, k, transpose_b=True)
        
        # Scale by square root of key dimension (prevents softmax saturation)
        d_k = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_logits = matrix_mult / tf.math.sqrt(d_k)

        # Apply mask (if provided) by adding large negative values to masked positions
        if mask is not None:
            scaled_logits += (mask * -1e9)

        # Apply softmax to get attention probabilities
        attention_weights = tf.nn.softmax(scaled_logits, axis=-1)
        
        # Apply attention weights to values
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, q, k, v, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            q: Query input
            k: Key input
            v: Value input
            mask: Optional attention mask
            
        Returns:
            Output after multi-head attention and linear projection
        """
        batch_size = tf.shape(q)[0]

        if logger_attention.isEnabledFor(logging.DEBUG):
            logger_attention.debug("      MHA: Q=%s, K=%s, V=%s", 
                                  q.shape, k.shape, v.shape)
            if mask is not None:
                logger_attention.debug("      Mask applied: %s", mask.shape)

        # Linear projections for Q, K, V
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split into multiple heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Apply attention function
        scaled_attention, attention_weights = self.attention_function(q, k, v, mask)

        if logger_attention.isEnabledFor(logging.DEBUG):
            if tf.executing_eagerly():
                # Average attention weright for all heads
                mean_attention = tf.reduce_mean(attention_weights).numpy()
                logger_attention.debug("      Attention weights: mean=%.4f, shape=%s", 
                                    mean_attention, attention_weights.shape)
            else:
                logger_attention.debug("      Attention weights shape: %s", 
                                      attention_weights.shape)

        # Transpose back: (batch_size, num_heads, seq_len, depth) -> (batch_size, seq_len, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        # Concatenate heads: (batch_size, seq_len, num_heads, depth) -> (batch_size, seq_len, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Final linear projection
        output = self.dense(concat_attention)

        return output


class Feedforward(keras.layers.Layer):
    """
    Position-wise Feed-Forward Network.
    
    A simple two-layer fully connected network applied to each position independently.
    This adds non-linearity and helps the model learn complex patterns.
    
    Architecture: Linear -> ReLU -> Linear
    
    Args:
        d_model: Output dimension (matches model dimension)
        d_ff: Hidden layer dimension (typically 4x d_model)
    """
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.dense1 = Dense(d_ff, activation='relu')  # Expansion layer
        self.dense2 = Dense(d_model)  # Projection back to d_model

        logger_init.info("Feedforward initialized")
        logger_init.info("FeedforwardlConfig: d_model=%s, d_ff=%s", 
                         d_model, d_ff)

    def call(self, x):
        """
        Forward pass through feed-forward network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor of same shape as input
        """

        x = self.dense1(x)  # Expand and apply ReLU
        output = self.dense2(x)  # Project back to original dimension

        return output


class Encoder(keras.layers.Layer):
    """
    Transformer Encoder Layer.
    
    Processes the input sequence through self-attention and feed-forward networks.
    Each encoder layer consists of:
        1. Multi-head self-attention
        2. Add & Normalize (residual connection + layer normalization)
        3. Feed-forward network
        4. Add & Normalize (residual connection + layer normalization)
    
    Args:
        d_model: Model embedding dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward network hidden dimension
        dropout_rate: Dropout probability for regularization
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Sub-layer 1: Multi-head self-attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Sub-layer 2: Feed-forward network
        self.feedforward = Feedforward(d_model, d_ff)

        # Layer normalization for stabilizing training
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        # Dropout for regularization
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

        logger_init.info("Encoder initialized")
        logger_init.info("EncoderlConfig: d_model=%s, heads=%s, d_ff=%s, dropout=%s", # Поменять стиль вывода 
                         d_model, num_heads, d_ff, dropout_rate)

    def call(self, x, training: bool = False, padding_mask=None):
        """
        Forward pass through encoder layer.
        
        Args:
            x: Input tensor
            training: Whether in training mode (affects dropout)
            padding_mask: Mask for padded positions (1 = valid, 0 = padding)
            
        Returns:
            Encoded representation
        """

        # logger_forward.debug("EncoderInput: x_shape=%s, training=%s, padding_mask=%s",
        #                      x.shape, training, padding_mask if padding_mask is not None else None)

        if logger_forward.isEnabledFor(logging.DEBUG):
            mask_status = "with padding mask" if padding_mask is not None else "no mask"
            logger_forward.debug("    Encoder layer input: %s (%s)", x.shape, mask_status)

        # Invert padding mask for attention (attention expects 0 = valid, 1 = masked)
        if padding_mask is not None:
            inverted_padding_mask = 1 - padding_mask
        else:
            inverted_padding_mask = None

        # Sub-layer 1: Self-attention
        attention_output = self.attention(x, x, x, inverted_padding_mask)
        attention_output = self.dropout1(attention_output, training=training)
        # Residual connection + layer normalization
        attention_output = self.layernorm1(x + attention_output)

        # Sub-layer 2: Feed-forward
        feedforward_output = self.feedforward(attention_output)
        feedforward_output = self.dropout2(feedforward_output, training=training)
        # Residual connection + layer normalization
        output = self.layernorm2(attention_output + feedforward_output)

        if logger_forward.isEnabledFor(logging.DEBUG):
            logger_forward.debug("    Encoder layer output: %s", output.shape)

        return output


class Decoder(keras.layers.Layer):
    """
    Transformer Decoder Layer.
    
    Processes the target sequence with attention to both itself and the encoder output.
    Each decoder layer consists of:
        1. Masked multi-head self-attention (prevents looking ahead)
        2. Add & Normalize
        3. Multi-head cross-attention to encoder output
        4. Add & Normalize
        5. Feed-forward network
        6. Add & Normalize
    
    Args:
        d_model: Model embedding dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward network hidden dimension
        dropout_rate: Dropout probability for regularization
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Sub-layer 1: Masked self-attention (decoder attends to itself)
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # Sub-layer 2: Cross-attention (decoder attends to encoder output)
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        
        # Sub-layer 3: Feed-forward network
        self.feedforward = Feedforward(d_model, d_ff)

        # Dropout layers for regularization
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

        # Layer normalization layers
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        logger_init.info("Decoder initialized")
        logger_init.info("DecoderlConfig: d_model=%s, heads=%s, d_ff=%s, dropout=%s",
                         d_model, num_heads, d_ff, dropout_rate)

    def call(self, x, enc_output=None, training: bool = False, attention_mask=None, padding_mask=None):
        """
        Forward pass through decoder layer.
        
        Args:
            x: Decoder input tensor
            enc_output: Encoder output (for cross-attention)
            training: Whether in training mode
            attention_mask: Causal mask to prevent attending to future positions
            padding_mask: Mask for padded positions
            
        Returns:
            Decoded representation
        """

        # logger_forward.debug("DecoderInput: x_shape=%s, enc_output=%s, training=%s, attention_mask=%s, padding_mask=%s",
        #                      x.shape, enc_output if enc_output is not None else None, training, attention_mask if attention_mask is not None else None, padding_mask if padding_mask is not None else None)

        if logger_forward.isEnabledFor(logging.DEBUG):
            logger_forward.debug("    Decoder layer input: %s", x.shape)
            if enc_output is not None:
                logger_forward.debug("    Cross-attention from encoder: %s", enc_output.shape)

        # Combine attention mask and padding mask
        inverted_universal_mask = None
        
        if attention_mask is not None and padding_mask is not None:
            universal_mask = attention_mask * padding_mask
            inverted_universal_mask = 1 - universal_mask
            if logger_forward.isEnabledFor(logging.DEBUG):
                logger_forward.debug("    Using combined mask (attention + padding)")
        
        elif attention_mask is not None:
            universal_mask = attention_mask
            inverted_universal_mask = 1 - universal_mask
            if logger_forward.isEnabledFor(logging.DEBUG):
                logger_forward.debug("    Using attention mask only")
        
        elif padding_mask is not None:
            universal_mask = padding_mask
            inverted_universal_mask = 1 - universal_mask
            if logger_forward.isEnabledFor(logging.DEBUG):
                logger_forward.debug("    Using padding mask only")
        
        else:
            if logger_forward.isEnabledFor(logging.DEBUG):
                logger_forward.debug("    No mask applied")

        # Sub-layer 1: Masked self-attention
        attention_output = self.attention(x, x, x, inverted_universal_mask)
        attention_output = self.dropout1(attention_output, training=training)
        attention_output = self.layernorm1(x + attention_output)
        
        # If no encoder output provided, use self-attention output
        if enc_output is None:
            enc_output = attention_output
        
        # Sub-layer 2: Cross-attention to encoder
        enc_dec_attention_output = self.enc_dec_attention(attention_output, enc_output, enc_output)
        enc_dec_attention_output = self.dropout2(enc_dec_attention_output, training=training)
        enc_dec_attention_output = self.layernorm2(attention_output + enc_dec_attention_output)

        # Sub-layer 3: Feed-forward
        feedforward_output = self.feedforward(enc_dec_attention_output)
        feedforward_output = self.dropout3(feedforward_output, training=training)
        output = self.layernorm3(enc_dec_attention_output + feedforward_output)

        if logger_forward.isEnabledFor(logging.DEBUG):
            logger_forward.debug("    Decoder layer output: %s", output.shape)

        return output


class Transformer(Model):
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
                 output_vocab_size: int, seq_len: int, dropout_rate: float = 0.1, input_vocab_size: int = None, 
                 target_vocab_size: int = None, use_positional_encoding=True, make_embadings=False):
        super().__init__()

        self.d_model = d_model
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.output_vocab_size = output_vocab_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.use_positional_encoding = use_positional_encoding
        
        # Generate positional encodings
        self.pos_encoding = positional_encoding(d_model, seq_len) 
        self.dropout = Dropout(dropout_rate)
        
        # Create embedding layers if vocabulary sizes are provided
        if input_vocab_size is not None:
            self.encoder_embedding = Embedding(input_vocab_size, d_model)

        if target_vocab_size is not None:
            self.decoder_embedding = Embedding(target_vocab_size, d_model)

        # Stack of encoder layers
        self.encoder = [Encoder(d_model, num_heads, d_ff, dropout_rate) 
                       for _ in range(num_encoder_layers)]

        # Stack of decoder layers
        self.decoder = [Decoder(d_model, num_heads, d_ff, dropout_rate) 
                       for _ in range(num_decoder_layers)]

        # Final linear layer to project to output vocabulary
        self.linear = Dense(output_vocab_size, use_bias=True)

        self._model_summary()
    
    def build(self, input_shape):
        super().build(input_shape)

    def call(self, args, training: bool = False):
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
            logger_forward.debug("  Mode: %s", "training" if training else "inference")
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

            seq_len = tf.shape(x_encoder)[1]

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
            x_encoder *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

            # Add positional encodings
            if self.use_positional_encoding:
                x_encoder += self.pos_encoding[:, :seq_len, :]
   
            # Apply dropout
            x_encoder = self.dropout(x_encoder, training=training)

            # Pass through all encoder layers
            for i, layer in enumerate(self.encoder):
                if logger_forward.isEnabledFor(logging.DEBUG):
                    logger_forward.debug("  Encoder layer %d/%d:", i+1, len(self.encoder))
                x_encoder = layer(x_encoder, training=training, padding_mask=encoder_padding_mask)

            # Logging encoder output
            if logger_forward.isEnabledFor(logging.DEBUG):
                logger_forward.debug("  Encoder output: %s", x_encoder.shape)
        
        # ===== DECODER PROCESSING =====
        if self.num_decoder_layers > 0:

            if logger_forward.isEnabledFor(logging.DEBUG):
                logger_forward.debug("Processing decoder (%d layers)...", self.num_decoder_layers)

            seq_len = tf.shape(x_decoder)[1]

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
            x_decoder *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

            if self.use_positional_encoding:
                x_decoder += self.pos_encoding[:, :seq_len, :]

            # Apply dropout
            x_decoder = self.dropout(x_decoder, training=training)

            # Pass through all decoder layers
            for i, layer in enumerate(self.decoder):
                if logger_forward.isEnabledFor(logging.DEBUG):
                    logger_forward.debug("  Decoder layer %d/%d:", i+1, len(self.decoder))
                x_decoder = layer(x_decoder, x_encoder, training=training, 
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
                              x_encoder.shape, output.shape,
                              "train" if training else "eval")

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
    

if __name__ == "__main__":
    """
    Example usage: Training a simple encoder-only Transformer for sequence classification.
    
    This example demonstrates:
        - Creating a Transformer with 2 encoder layers and no decoder
        - Training on dummy data with pre-computed embeddings
        - Using the model for sequence-level predictions
    """

    from logger import setup_logging

    setup_logging()

    logging.getLogger(f'{__name__}.init').setLevel(logging.INFO)
    logging.getLogger(f'{__name__}.forward').setLevel(logging.WARNING) 
    logging.getLogger(f'{__name__}.attention').setLevel(logging.WARNING)

    # ===== MODEL CONFIGURATION =====
    num_encoders = 2  # Number of encoder layers
    num_decoders = 0  # Number of decoder layers (0 = encoder-only model)
    d_model = 296  # Embedding dimension
    num_heads = 8  # Number of attention heads
    d_ff = 128  # Feed-forward hidden dimension
    output_vocab_size = 2  # Binary classification
    seq_len = 40  # Sequence length

    # ===== TRAINING CONFIGURATION =====
    BATCH_SIZE = 32
    EPOCHS = 5  

    # Example causal/look-ahead mask (for autoregressive tasks)
    # Each row shows which positions a token can attend to
    attention_mask = [
        [1, 0, 0, 0, 0, 0, 0],  # First token can only see itself
        [1, 1, 0, 0, 0, 0, 0],  # Second token can see first two tokens
        [1, 1, 1, 0, 0, 0, 0],  # Third token can see first three tokens
        [1, 1, 1, 1, 0, 0, 0],  # And so on...
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1]   # Last token can see all tokens
    ]

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
    
    # Compile model with optimizer and loss function
    transformer.compile(
        optimizer='adam', 
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),  # from_logits=True since model outputs raw scores
        metrics=['accuracy'],
        weighted_metrics=['accuracy']
    )
    
    transformer.name = "Test_model"

    # Enable eager execution for debugging
    tf.config.run_functions_eagerly(True)

    # ===== DUMMY DATA GENERATION =====
    # Create random input data (pre-computed embeddings)
    x_train = np.ones((175, seq_len, d_model))
    x_test = np.ones((59, seq_len, d_model))
    
    # Create one-hot encoded labels
    y_train = np.zeros((175, seq_len, output_vocab_size))
    y_test = np.zeros((59, seq_len, output_vocab_size))

    # Populate labels with cyclic pattern
    for i, val in enumerate(y_train):
        for j, el in enumerate(val):
            y_train[i][j][j % output_vocab_size] = 1

    for i, val in enumerate(y_test):
        for j, el in enumerate(val):
            y_test[i][j][j % output_vocab_size] = 1
    
    # ===== PREPARE TRAINING DATA =====
    # Model expects dictionary input with encoder_input and decoder_input
    input = {
        "encoder_input": x_train,
        "decoder_input": np.zeros((len(x_train), seq_len, d_model)),  # Dummy decoder input
    }

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((input, y_train))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # ===== PREPARE VALIDATION DATA =====
    validation_input = {
        "encoder_input": x_test,
        "decoder_input": np.zeros((len(x_test), seq_len, d_model)),
    }

    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_input, y_test))
    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    # ===== TRAIN MODEL =====
    history = transformer.fit(
        dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset
    )