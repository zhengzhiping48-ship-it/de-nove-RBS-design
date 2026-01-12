import pandas as pd
import tensorflow as tf
from keras import layers
from keras.callbacks import EarlyStopping
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, KBinsDiscretizer
from sklearn.pipeline import Pipeline, make_pipeline
import datetime

# ignore warnings
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

seq_length = 25
num_layers = 2
d_model = 128
num_heads = 8
dff = 512
batch_size = 128


def cutWord(x, window=1, length=seq_length):
    seq2word = []
    for i in range(length):
        seq2word.append(x[i * window:i * window + window])
    return " ".join(seq2word)


x_test_df = pd.read_csv('./W_GAN_4999_7.txt', header=None, names=['seq'])

x_test_spaced = x_test_df.applymap(cutWord)
x_test_spaced_array = x_test_spaced["seq"].to_numpy()
vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=d_model,
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    ngrams=None,
    output_mode='int'
)
vectorize_layer.adapt(x_test_spaced_array)

x_test_vectorized = vectorize_layer(x_test_spaced_array)


def RKA_bin(x):
    if x >= 0.8:
        return 1
    else:
        return 0


def MFE_bin(x):
    if x >= -14.:
        return 1
    else:
        return 0


def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # Shape = (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    # calculate matmul_qk_v
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights, matmul_qk


# create multi-head attention layer
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # Because for multi-head, head number * depth = multi-head
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        # Set layers for q, k, v
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # shape = (batch_size, num_heads, seq_len, depth)

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights, matmul_qk = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights, matmul_qk


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        # define layers
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        attn_output, attn_weight, matmul_qk = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attn_weight, matmul_qk


def get_angles(pos, i, d_model):
    angle_rate = 1 / np.power(10000, (2 * (i / 2)) / np.float32(d_model))
    return pos * angle_rate


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.3):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers  # how many encoder layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.attention_weights = {}
        self.matmul_qks = {}

    def call(self, x, training, mask=None):
        # encoding and position encoding
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block, matmul_qk = self.enc_layers[i](x, training, mask)
            self.attention_weights[f'attentions_{i + 1}'] = block
            self.matmul_qks[f'matmul_qk_{i + 1}'] = matmul_qk

        return x, block  # (batch_size, input_seq_len, d_model)

    def get_attention(self):
        return self.attention_weights

    def get_matmul_qks(self):
        return self.matmul_qks


sample_encoder = Encoder(num_layers=2, d_model=5, num_heads=1,
                         dff=dff, input_vocab_size=10,
                         maximum_position_encoding=30)
temp_input = vectorize_layer(x_test_spaced)
sample_encoder_output = sample_encoder(temp_input[22:29], training=False, mask=None)
print(sample_encoder_output[0].shape)


def create_model(seq_length, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding,
                 rate=0.3):
    input = tf.keras.Input(shape=(seq_length,))
    x, aws = Encoder(num_layers, d_model, num_heads, dff,
                     input_vocab_size, maximum_position_encoding, rate=rate)(input)
    x = tf.keras.layers.Reshape((seq_length * d_model,))(x)
    x = tf.keras.layers.Dense(seq_length, activation='relu')(x)
    x = tf.keras.layers.Dense(1)(x)
    output = tf.squeeze(x)

    return tf.keras.Model(inputs=input, outputs=output)


model = create_model(seq_length=seq_length, num_layers=num_layers, d_model=d_model, num_heads=num_heads,
                     dff=dff, input_vocab_size=10,
                     maximum_position_encoding=30)
p = vectorize_layer(x_test_spaced.iloc[17])
print(model(p))

# Load the saved model
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from keras.models import load_model
from keras.utils import get_custom_objects

get_custom_objects().update({'Encoder': Encoder})

model = load_model('./checkpoints_64_21_pos_copy/model_2_8_128_512/model')

x_test_df = pd.read_csv('./W_GAN_49999.txt', header=None, names=['seq'])

x_test_spaced = x_test_df.applymap(cutWord)
x_test_spaced_array = x_test_spaced["seq"].to_numpy()

vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=d_model,
    standardize='lower_and_strip_punctuation',
    split='whitespace',
    ngrams=None,
    output_mode='int'
)
vectorize_layer.adapt(x_test_spaced_array)

x_test_vectorized = vectorize_layer(x_test_spaced)

print(f"x_test_vectorized shape: {x_test_vectorized.shape}")

x_test_dataset = tf.data.Dataset.from_tensor_slices(x_test_vectorized)
batch_size = 32
x_test_dataset = x_test_dataset.batch(batch_size)

# Print model layers and weights
for layer in model.layers:
    if len(layer.trainable_weights) == 0:
        print(f"Layer {layer.name} has no trainable weights.")
    else:
        print(f"Layer {layer.name} trainable weights: {layer.trainable_weights}")


def generate_saliency_map(model, input_tensor):
    input_tensor = tf.cast(input_tensor, dtype=tf.float32)
    input_tensor = tf.Variable(input_tensor)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor)
        loss = tf.reduce_mean(predictions)

    gradients = tape.gradient(loss, input_tensor)

    if gradients is None:
        raise ValueError("Gradient is None, unable to compute saliency map.")

    saliency_map = tf.abs(gradients).numpy().squeeze()
    return saliency_map


# Try to generate saliency map for first batch
for batch in x_test_dataset:
    input_tensor = batch
    print(f"Input tensor shape: {input_tensor.shape}")

    try:
        saliency_map = generate_saliency_map(model, input_tensor)
        print("Saliency Map:", saliency_map)

        plt.figure(figsize=(10, 1))
        plt.imshow(saliency_map[np.newaxis, :], aspect="auto", cmap='hot')
        plt.colorbar(label='Saliency')
        plt.title('Saliency Map')
        plt.show()
    except ValueError as e:
        print(f"Error generating saliency map: {e}")

    break

# Make predictions
predictions = model.predict(x_test_dataset)
print(predictions)

# Save predictions to CSV
pd.DataFrame(predictions, columns=["prediction"]).to_csv("predictions.csv", index=False)
print("Predictions saved to predictions.csv")