# We set the backend to TensorFlow. The code works with
# both `tensorflow` and `torch`. It does not work with JAX
# due to the behavior of `jax.numpy.tile` in a jit scope
# (used in `TransformerDecoder.get_causal_attention_mask()`:
# `tile` in JAX does not support a dynamic `reps` argument.
# You can make the code work in JAX by wrapping the
# inside of the `get_causal_attention_mask` method in
# a decorator to prevent jit compilation:
# `with jax.ensure_compile_time_eval():`.
import os
import sys

os.environ["KERAS_BACKEND"] = "tensorflow"

import pathlib
import random
import string
import re
import numpy as np

import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
import tensorflow as tf

import keras
from keras import layers
from keras import ops
from keras.layers import TextVectorization

from custom_multihead_attention_layer import CustomMultiHeadAttention

current_dir = os.getcwd()

# text_file = keras.utils.get_file(
#     fname="spa-eng.zip",
#     origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
#     extract=True,
#     cache_dir=current_dir,  # Set the base cache directory to your project folder
#     cache_subdir="."        # Tell it not to create an extra 'datasets' folder
# )

text_file = pathlib.Path(current_dir) / "assignment-2" / "spa-eng" / "spa.txt"

with open(text_file, encoding='utf-8') as f:
    lines = f.read().split("\n")[:-1]
    
text_pairs = []
for line in lines:
    eng, spa = line.split("\t")
    eng = "[start] " + eng + " [end]"
    text_pairs.append((spa, eng))

for _ in range(5):
    print(random.choice(text_pairs))

random.shuffle(text_pairs)

# Only work with a subset of the data
subset_size = 20000
text_pairs = text_pairs[:subset_size]

num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = 15000
sequence_length = 20
batch_size = 64

class CustomLogSaver(keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        # Create or clear the file when training starts
        with open(self.filepath, 'w') as f:
            pass 

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Format the string exactly as you requested
        log_string = (
            f"accuracy: {logs.get('accuracy', 0):.4f} - "
            f"loss: {logs.get('loss', 0):.4f} - "
            f"val_accuracy: {logs.get('val_accuracy', 0):.4f} - "
            f"val_loss: {logs.get('val_loss', 0):.4f}\n"
        )
        
        # Append to the file
        with open(self.filepath, 'a') as f:
            f.write(log_string)

def custom_standardization(input_string):
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

eng_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
spa_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
train_eng_texts = [pair[0] for pair in train_pairs]
train_spa_texts = [pair[1] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
spa_vectorization.adapt(train_spa_texts)

def format_dataset(eng, spa):
    eng = eng_vectorization(eng)
    spa = spa_vectorization(spa)
    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": spa[:, :-1],
        },
        spa[:, 1:],
    )

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf_data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(2048).prefetch(16)

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = CustomMultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def build(self, input_shape):
        self.attention.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, mask=None, return_attention_scores=False):
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, :], dtype="int32")
        else:
            padding_mask = None

        # Conditionally retrieve the attention scores
        if return_attention_scores:
            attention_output, attention_scores = self.attention(
                inputs, padding_mask, return_attention_scores=True
            )
        else:
            attention_output = self.attention(inputs, padding_mask)

        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        final_output = self.layernorm_2(proj_input + proj_output)
        
        if return_attention_scores:
            return final_output, attention_scores
        return final_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = ops.shape(inputs)[-1]
        positions = ops.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return ops.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


import keras
from keras import layers, ops

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.supports_masking = True

        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            name="self_attention",
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            name="cross_attention",
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ],
            name="dense_proj",
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

    def build(self, input_shape):
        # input_shape should be [decoder_shape, encoder_shape]
        decoder_shape = input_shape[0]
        encoder_shape = input_shape[1]

        self.attention_1.build(decoder_shape, decoder_shape, decoder_shape)
        self.attention_2.build(decoder_shape, encoder_shape, encoder_shape)
        self.dense_proj.build(decoder_shape)
        self.layernorm_1.build(decoder_shape)
        self.layernorm_2.build(decoder_shape)
        self.layernorm_3.build(decoder_shape)

        super().build(input_shape)

    def call(self, inputs, mask=None):
        decoder_inputs = inputs[0]
        encoder_outputs = inputs[1]

        causal_mask = self.get_causal_attention_mask(decoder_inputs)

        decoder_padding_mask = None
        encoder_padding_mask = None

        if mask is not None:
            if isinstance(mask, (list, tuple)):
                if len(mask) > 0:
                    decoder_padding_mask = mask[0]
                if len(mask) > 1:
                    encoder_padding_mask = mask[1]
            else:
                decoder_padding_mask = mask

        attn_output_1 = self.attention_1(
            query=decoder_inputs,
            value=decoder_inputs,
            key=decoder_inputs,
            attention_mask=causal_mask,
            query_mask=decoder_padding_mask,
            value_mask=decoder_padding_mask,
            key_mask=decoder_padding_mask,
        )
        out_1 = self.layernorm_1(decoder_inputs + attn_output_1)

        attn_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            query_mask=decoder_padding_mask,
            value_mask=encoder_padding_mask,
            key_mask=encoder_padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attn_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        import tensorflow as tf

        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        i = tf.range(seq_len)[:, None]
        j = tf.range(seq_len)[None, :]
        mask = tf.cast(i >= j, dtype=tf.bool)

        mask = tf.expand_dims(mask, axis=0)
        mask = tf.broadcast_to(mask, [batch_size, seq_len, seq_len])
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

embed_dim = 256
latent_dim = 2048
num_heads = 8

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim, name="encoder_embeddings")(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads, name="transformer_encoder")(x)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")

x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)([x, encoded_seq_inputs])
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)

decoder = keras.Model(
    [decoder_inputs, encoded_seq_inputs],
    decoder_outputs,
    name="decoder",
)

final_decoder_outputs = decoder([decoder_inputs, encoder_outputs])

transformer = keras.Model(
    {"encoder_inputs": encoder_inputs, "decoder_inputs": decoder_inputs},
    final_decoder_outputs,
    name="transformer",
)

epochs = 50  # This should be at least 30 for convergence

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
    CustomLogSaver("exp-1.txt") # This will save to your project root
]

transformer.summary()
transformer.compile(
    "rmsprop",
    loss=keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
    metrics=["accuracy"],
)
transformer.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)