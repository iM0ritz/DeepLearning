import keras
from keras import layers
from keras import ops

class CustomMultiHeadAttention(layers.Layer):

    # TODO: start out with simple word embeddings
    # Initialize a matrix for "Query" and "Key" (how big are they exactly? i think they need to have a n*n size where n is the length of a token embedding vector.)
    # Create Query and Key Vector for every token by taking multiplying token embedding and the matrix
    # for every query and every key, take dot product to see how much they align and divide result by sqrt(dimension of the query key space)
    # iterate through each token query and create softmax over all token keys.
    # Create Value vectors for every token using the Value Matrix.
    # For value resulting value vectors with the softmax values for every token
    # for every token, compute sum of updated softmax (query-key) matrix.
    # add sum to the current token embedding

    # we need to make use of cross-attention, because we have two completely different texts (spanish and english)
    # Queries should come from english and keys from spanish

    # for every head, we then once again add the sum of all attention heads to the embedding
    # if we have N heads, the matrices will all be reduced from dimensionality d to d/N.

    # This layer first projects query, key and value. These are (effectively) a list of tensors of length num_attention_heads, where the corresponding shapes are (batch_size, <query dimensions>, key_dim), (batch_size, <key/value dimensions>, key_dim), (batch_size, <key/value dimensions>, value_dim).

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

    def call(self, query, value, key):
        pass

    def compute_attention_scores(self, instance):
        pass
