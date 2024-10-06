#!/usr/bin/env python3
"""
Transformer app project
inclues positional encoding de attention project

By Ced
"""
import numpy as np
import tensorflow as tf

def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer
    param max_seq_len: integer representing the maximum sequence length
    param dm: integer representing the model depth
    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing
    the positional encoding vectors
    """

    PE = np.zeros((max_seq_len, dm))

    # Calcul de chaque position et dimension
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            # Calcul des valeurs de sinus et cosinus
            angle = pos / np.power(10000, (2 * (i // 2)) / dm)
            PE[pos, i] = np.sin(angle)
            if i + 1 < dm:
                PE[pos, i + 1] = np.cos(angle)

    return PE



def sdp_attention(Q, K, V, mask=None):
    """
    Scaled Dot Product Attention is a key component of
    the transformer architecture
    param Q: tensor with shape (..., seq_len_q, dk) containing the query matrix
    param K: tensor with shape (..., seq_len_v, dk) containing the key matrix
    param V: tensor with shape (..., seq_len_v, dv) containing the value matrix
    param mask is always None
    Returns: output, weights
        outputa tensor with its last two dimensions as (..., seq_len_q, dv)
            containing scaled dot product attention
        weights a tensor with its last two dimensions as
            (..., seq_len_q, seq_len_v) containing the attention weights
    """

    # Calcul du produit scalaire entre Q et K transposé
    matmul_QK = tf.matmul(Q, K, transpose_b=True)

    # Mise à l'échelle par la racine carrée de la dimension des clés
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_scores = matmul_QK / tf.math.sqrt(dk)

    # Appliquer un masque (optionnel)
    if mask is not None:
        scaled_scores += (mask * -1e9)  # -1e9: -1*10^9 for softmax

    # Appliquer Softmax aux scores pour obtenir les poids d'attention
    attention_weights = tf.nn.softmax(scaled_scores, axis=-1)

    # Multiplication des poids d'attention avec les valeurs V
    output = tf.matmul(attention_weights, V)

    return output, attention_weights



class MultiHeadAttention(tf.keras.layers.Layer):
    """
    perform multi head attention:
    """

    def __init__(self, dm, h):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def reshape_tensor(self, x, heads, flag):
        """
        reshapes the output of the attention block
        in order to calculate the output of the multi-head attention
        """
        if flag:
            # Tensor shape after reshaping and transposing:
            # (batch_size, heads, seq_length, -1)
            x = tf.reshape(x, shape=(tf.shape(x)[0],
                                     tf.shape(x)[1], self.h, -1))
            x = tf.transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations:
            # (batch_size, seq_length, d_k)
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], self.dm))
        return x

    def call(self, Q, K, V, mask=None):
        """
        Call method, return 2 values
        """
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.Wq(Q), self.h, True)
        k_reshaped = self.reshape_tensor(self.Wk(K), self.h, True)
        v_reshaped = self.reshape_tensor(self.Wv(V), self.h, True)
        o_reshaped, attention_W = sdp_attention(q_reshaped,
                                                k_reshaped,
                                                v_reshaped,
                                                mask)

        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.h, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)

        # Apply one final linear projection to the output to generate
        # the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.linear(output), attention_W



class EncoderBlock(tf.keras.layers.Layer):
    """
    Encoder block for a transformer
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.\
            LayerNormalization(axis=-1, epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.\
            LayerNormalization(axis=-1, epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        x is tensor of shape (batch, input_seq_len, dm)
        training is a boolean to determine if the model is training
        mask is...
        returns: tensor of shape (batch, input_seq_len, dm)
        containning the block’s output
        """

        out1, Attention_W = self.mha(x, x, x, mask)
        drop1 = self.dropout1(out1, training=training)
        out2 = self.layernorm1(x + drop1)
        out3 = self.dense_hidden(out2)
        out4 = self.dense_output(out3)

        drop2 = self.dropout2(out4, training=training)
        decoder_out = self.layernorm2(out2 + drop2)

        return decoder_out





class DecoderBlock(tf.keras.layers.Layer):
    """
    This class do the decoder part of the transformer
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.\
            LayerNormalization(axis=-1, epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.\
            LayerNormalization(axis=-1, epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.\
            LayerNormalization(axis=-1, epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        call function,
        returns: tensor of shape (batch, target_seq_len, dm)
        """

        out1, _ = self.mha1(x, x, x, padding_mask)
        drop1 = self.dropout1(out1, training=training)
        norm1 = self.layernorm1(x + drop1)

        # decoder output into MHA_2
        out3, _ = self.mha2(norm1, encoder_output,
                            encoder_output, look_ahead_mask)
        out3 = self.dropout2(out3, training=training)
        norm2 = self.layernorm2(out3 + norm1)

        # feed forward
        feed_input = self.dense_hidden(norm2)
        feed_output = self.dense_output(feed_input)
        feed_drop = self.dropout3(feed_output, training=training)

        decoder_output = self.layernorm3(norm2 + feed_drop)

        return decoder_output


class Encoder (tf.keras.layers.Layer):
    """
    This class create an encoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        super(Encoder, self).__init__()
        self.N = N  # number of blocks
        self.dm = dm  # dimensionality of the model
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Call method, return a tensor of
        shape (batch, input_seq_len, dm)
        containing the encoder output
        """
        seq_len = tf.shape(x)[1]

        # Word Embedding
        embedding = self.embedding(x)  # (batch, input_seq_len, dm)

        # Scale embedding
        embedding *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # Add positional encoding
        pos_encoding = self.positional_encoding[:seq_len, :]
        encoder_input = embedding + pos_encoding

        # Apply dropout
        x = self.dropout(encoder_input, training=training)

        # Pass through each encoder block
        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        return x



class Decoder(tf.keras.layers.Layer):
    """
    This class create a decoder for a transformer
    """
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        super(Decoder, self).__init__()
        self.dm = dm
        self.N = N
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        call method: return output of de decoder, same shape as x
        """
        embedding = self.embedding(x)
        embedding *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        positional_encoding = self.positional_encoding[:x.shape[1], :]
        x = embedding + positional_encoding
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, encoder_output,
                      training, look_ahead_mask,
                      padding_mask)
        return x



class Transformer(tf.keras.Model):
    """
    whole transformer model
    """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate=0.1)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_input,
                               drop_rate=0.1)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training,
             encoder_mask, look_ahead_mask, decoder_mask):
        """
        build the transformer model as described in the paper
        on calling, it should return the model of the transformer
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output,
                                      training, look_ahead_mask,
                                      decoder_mask)

        output = self.linear(decoder_output)
        return output
