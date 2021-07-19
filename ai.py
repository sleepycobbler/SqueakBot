import tensorflow as tf
from tensorflow.keras.layers.experimental import keras_preprocessing
from tensorflow.keras import layers
from tensorflow.keras.layers.preprocessing import text

keras.layers.experimental.preprocessing.StringLookup(
    max_tokens=None, num_oov_indices=1, mask_token=None,
    oov_token='[UNK]', vocabulary=None, encoding=None, invert=False,
    output_mode=index_lookup.INT, sparse=False, pad_to_max_tokens=False, **kwargs
)

import nltk
nltk.download('punkt')

import numpy as np
import os
import time
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
    await ctx.channel.send(ben_text.generate(length=random.randrange(5, 100), random_seed=random.randint(0, 9999999999999999999999)))

async def ai_gen_tf(ctx):
    data = filer.read()
    id = str(ctx.guild.id)
    text = "\n".join(data[id])

    vocab = sorted(set(text))

    example_texts = ['abcdefg', 'xyz']

    chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')

    ids_from_chars = tf.keras.preprocessing.text.text_to_word_sequence(vocabulary=list(vocab), mask_token=None)

    ids = ids_from_chars(chars)

    chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    chars = chars_from_ids(ids)

    tf.strings.reduce_join(chars, axis=-1).numpy()

    def text_from_ids(ids):
        return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    seq_length = 100
    examples_per_epoch = len(text)//(seq_length+1)

    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

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

    # Length of the vocabulary in chars
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024

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
    
    model = MyModel(
        vocab_size=len(ids_from_chars.get_vocabulary()),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    mean_loss = example_batch_loss.numpy().mean()

    tf.exp(mean_loss).numpy()

    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS = 20

    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    class OneStep(tf.keras.Model):
        def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
          super().__init__()
          self.temperature = temperature
          self.model = model
          self.chars_from_ids = chars_from_ids
          self.ids_from_chars = ids_from_chars

          # Create a mask to prevent "[UNK]" from being generated.
          skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
          sparse_mask = tf.SparseTensor(
              # Put a -inf at each bad index.
              values=[-float('inf')]*len(skip_ids),
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
          predicted_logits = predicted_logits/self.temperature
          # Apply the prediction mask: prevent "[UNK]" from being generated.
          predicted_logits = predicted_logits + self.prediction_mask

          # Sample the output logits to generate token IDs.
          predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
          predicted_ids = tf.squeeze(predicted_ids, axis=-1)

          # Convert from token ids to characters
          predicted_chars = self.chars_from_ids(predicted_ids)

          # Return the characters and model state.
          return predicted_chars, states

    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    start = time.time()
    states = None
    next_char = tf.constant(['ROMEO:'])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
    print('\nRun time:', end - start)

    start = time.time()
    states = None
    next_char = tf.constant(['ROMEO:', 'ROMEO:', 'ROMEO:', 'ROMEO:', 'ROMEO:'])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result, '\n\n' + '_'*80)
    print('\nRun time:', end - start)

    tf.saved_model.save(one_step_model, 'one_step')
    one_step_reloaded = tf.saved_model.load('one_step')

    states = None
    next_char = tf.constant(['ROMEO:'])
    result = [next_char]

    for n in range(100):
        next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
        result.append(next_char)

    print(tf.strings.join(result)[0].numpy().decode("utf-8"))

    class CustomTraining(MyModel):
        @tf.function
        def train_step(self, inputs):
            inputs, labels = inputs
            with tf.GradientTape() as tape:
                predictions = self(inputs, training=True)
                loss = self.loss(labels, predictions)
            grads = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
            return {'loss': loss}

    model = CustomTraining(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

    model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    model.fit(dataset, epochs=1)

    EPOCHS = 10

    mean = tf.metrics.Mean()

    for epoch in range(EPOCHS):
        start = time.time()

        mean.reset_states()
        for (batch_n, (inp, target)) in enumerate(dataset):
            logs = model.train_step([inp, target])
            mean.update_state(logs['loss'])

            if batch_n % 50 == 0:
                template = f"Epoch {epoch+1} Batch {batch_n} Loss {logs['loss']:.4f}"
                print(template)

        # saving (checkpoint) the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.save_weights(checkpoint_prefix.format(epoch=epoch))

        print()
        print(f'Epoch {epoch+1} Loss: {mean.result().numpy():.4f}')
        print(f'Time taken for 1 epoch {time.time() - start:.2f} sec')
        print("_"*80)

    model.save_weights(checkpoint_prefix.format(epoch=epoch))

    pass
