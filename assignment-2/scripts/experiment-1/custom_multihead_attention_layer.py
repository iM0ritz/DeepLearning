import keras
from keras import layers
from keras import ops

class CustomMultiHeadAttention(layers.Layer):

    def __init__(
        self,
        num_heads,
        key_dim,
        **kwargs):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.key_dim = key_dim

    def build(self, input_shape):
        # input shape is (batch, seq, embed) and we want to get the embedding dimension
        embed_dim = input_shape[-1]

        self.Wq = self.add_weight(
            name="Wq",
            shape=(embed_dim, embed_dim),
            initializer="glorot_uniform",
            trainable=True
        )

        self.Wk = self.add_weight(
            name="Wq",
            shape=(embed_dim, embed_dim),
            initializer="glorot_uniform",
            trainable=True
        )

        self.Wv = self.add_weight(
            name="Wq",
            shape=(embed_dim, embed_dim),
            initializer="glorot_uniform",
            trainable=True
        )
    
        self.Wo = self.add_weight(
            name="Wo",
            shape=(embed_dim, embed_dim),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        attention_output = self.compute_attention_scores(inputs)
        return attention_output
    
    def compute_attention_scores(self, inputs):
        batch_size = ops.shape(inputs)[0]

        q = ops.matmul(inputs, self.Wq) 
        k = ops.matmul(inputs, self.Wk)
        v = ops.matmul(inputs, self.Wv)

        # reshape matrices to calculate matmuls for all attention heads at the same time
        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # main logic from transformer paper
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        dk = self.key_dim / self.num_heads
        scaled_scores = scores / ops.sqrt(dk)
        weights = ops.softmax(scaled_scores)
        attention_output = ops.matmul(weights, v)

        # reshape back
        joined_output = self._join_heads(attention_output, batch_size)

        return ops.matmul(joined_output, self.Wo)

    def _split_heads(self, x, batch_size):
        # reshape from (batch_size, seq_len, embed_dim) to (batch_size, num_heads, seq_len, head_dim)
        head_dim = self.key_dim // self.num_heads
        x = ops.reshape(x, (batch_size, -1, self.num_heads, head_dim))
        return ops.transpose(x, (0, 2, 1, 3))
    
    def _join_heads(self, x, batch_size):
        # reshape from (batch_size, num_heads, seq_len, head_dim) back to (batch_size, seq_len, embed_dim)
        x = ops.transpose(x, (0, 2, 1,3 ))
        return ops.reshape(x, (batch_size, -1, self.key_dim))