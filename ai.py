import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

import nltk

nltk.download('punkt')

import random

import filer


async def ai_gen_nltk(ctx):
    data = filer.read()
    words = []
    for guild in data.keys():
        for message in data[guild]:
            if message != "" and not message.startswith('http') and not message.startswith('<'):
                message_alt = message
                for word in nltk.word_tokenize(message_alt):
                    words.append(word)
    ben_text = nltk.text.Text(words)
    await ctx.channel.send(
        ben_text.generate(length=random.randrange(5, 100), random_seed=random.randint(0, 9999999999999999999999)))


async def ai_gen_tf(ctx):
    data = filer.read()
    id = str(ctx.guild.id)
    text = "\n".join(data[id])

    vocab = sorted(set(text))
    print(vocab)
    example_texts = ['abcdefg', 'xyz']

    ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))
    chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(),
                                                                             invert=True)

    def text_from_ids(ids):
        return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    seq_length = 100

    sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    split_input_target(list("Tensorflow"))

    dataset = sequences.map(split_input_target)

    # Batch size
    BATCH_SIZE = 64

    BUFFER_SIZE = 10000

    dataset = (
        dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

    class MyModel(tf.keras.Model):
        def __init__(self, vocab_size, embedding_dim, rnn_units):
            super().__init__(self)
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
            self.gru = tf.keras.layers.GRU(rnn_units,
                                           return_sequences=True,
                                           return_state=True)
            self.dense = tf.keras.layers.Dense(vocab_size)

        def call(self, inputs, states=None, return_state=False, training=False):
            x = inputs
            x = self.embedding(x, training=training)
            if states is None:
                states = self.gru.get_initial_state(x)
            x, states = self.gru(x, initial_state=states, training=training)
            x = self.dense(x, training=training)

            if return_state:
                return x, states
            else:
                return x

    vocab_size = len(ids_from_chars.get_vocabulary())
    embedding_dim = 256
    rnn_units = 1024

    model = MyModel(
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    model.summary()

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)

    if os.path.isfile("my_model\{saved_model.pbtxt|saved_model.pb}"):
        class OneStep(tf.keras.Model):
            def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
                super().__init__()
                self.temperature = temperature
                self.model = model
                self.chars_from_ids = chars_from_ids
                self.ids_from_chars = ids_from_chars

                # Create a mask to prevent "" or "[UNK]" from being generated.
                skip_ids = self.ids_from_chars(['', '[UNK]'])[:, None]
                sparse_mask = tf.SparseTensor(
                    # Put a -inf at each bad index.
                    values=[-float('inf')] * len(skip_ids),
                    indices=skip_ids,
                    # Match the shape to the vocabulary
                    dense_shape=[len(ids_from_chars.get_vocabulary())])
                self.prediction_mask = tf.sparse.to_dense(sparse_mask)

        model = keras.models.load_model("my_model")

        one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

        # Now, let's generate a 1000 character chapter by giving our model "Chapter 1"
        # as its starting text
        states = None
        next_char = tf.constant(['Dude'])
        result = [next_char]

        for n in range(1000):
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)

        result = tf.strings.join(result)

        # Print the results formatted.
        print(result[0].numpy().decode('utf-8'))
        await ctx.channel.send(result[0].numpy().decode('utf-8'))
        return

    history = model.fit(dataset, epochs=20)

    model.save("model_one")

    class OneStep(tf.keras.Model):
        def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
            super().__init__()
            self.temperature = temperature
            self.model = model
            self.chars_from_ids = chars_from_ids
            self.ids_from_chars = ids_from_chars

            # Create a mask to prevent "" or "[UNK]" from being generated.
            skip_ids = self.ids_from_chars(['', '[UNK]'])[:, None]
            sparse_mask = tf.SparseTensor(
                # Put a -inf at each bad index.
                values=[-float('inf')] * len(skip_ids),
                indices=skip_ids,
                # Match the shape to the vocabulary
                dense_shape=[len(ids_from_chars.get_vocabulary())])
            self.prediction_mask = tf.sparse.to_dense(sparse_mask)

        @tf.function
        def generate_one_step(self, inputs, states=None):
            # Convert strings to token IDs.
            input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
            input_ids = self.ids_from_chars(input_chars).to_tensor()

            # Run the model.
            # predicted_logits.shape is [batch, char, next_char_logits]
            predicted_logits, states = self.model(inputs=input_ids, states=states,
                                                  return_state=True)
            # Only use the last prediction.
            predicted_logits = predicted_logits[:, -1, :]
            predicted_logits = predicted_logits / self.temperature

            # Apply the prediction mask: prevent "" or "[UNK]" from being generated.
            predicted_logits = predicted_logits + self.prediction_mask

            # Sample the output logits to generate token IDs.
            predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
            predicted_ids = tf.squeeze(predicted_ids, axis=-1)

            # Return the characters and model state.
            return chars_from_ids(predicted_ids), states

    # Create an instance of the character generator
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    # Now, let's generate a 1000 character chapter by giving our model "Chapter 1"
    # as its starting text
    states = None
    next_char = tf.constant(['Chapter 1'])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)

    # Print the results formatted.
    print(result[0].numpy().decode('utf-8'))
    await ctx.channel.send(result[0].numpy().decode('utf-8'))
