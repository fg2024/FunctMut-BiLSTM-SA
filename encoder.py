# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf


# "Attention is all you need" 中的编码模块
class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layers, d_model, num_heads, middle_units,
                 max_seq_len, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.n_layers = n_layers
        self.d_model = d_model
        self.pos_embedding = PositionalEncoding(sequence_len=max_seq_len, embedding_dim=d_model)

        self.encode_layer = [EncoderLayer(d_model=d_model, num_heads=num_heads,
                                          middle_units=middle_units) for _ in range(n_layers)]

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"n_layers": self.n_layers, "d_model": self.d_model,
                  "pos_embedding": self.pos_embedding, "encode_layer": self.encode_layer}
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        emb = self.pos_embedding(inputs)

        for i in range(self.n_layers):
            emb = self.encode_layer[i](emb)

        return emb


# 位置编码
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None, **kwargs):
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        super(PositionalEncoding, self).__init__(**kwargs)

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"sequence_len": self.sequence_len, "embedding_dim": self.embedding_dim}
        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        if self.embedding_dim is None:
            self.embedding_dim = int(inputs.shape[-1])

        position_embedding = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
            for pos in range(self.sequence_len)])

        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # dim 2i
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # dim 2i+1

        position_embedding = tf.cast(position_embedding, dtype=tf.float32)

        return position_embedding + inputs


# 编码层
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, middle_units, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, middle_units)

        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"mha": self.mha, "ffn": self.ffn,
                  "layernorm1": self.layernorm1, "layernorm2": self.layernorm2}
        base_config = super(EncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, **kwargs):
        # 多头注意力网络
        att_output = self.mha([inputs, inputs, inputs])
        out1 = self.layernorm1(inputs + att_output)  # (batch_size, input_seq_len, d_model)

        # 前向网络
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


# 构造 multi head attention 层
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0

        # 分头后的维度
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

        self.dot_attention = scaled_dot_product_attention

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"num_heads": self.num_heads, "d_model": self.d_model,
                  "depth": self.depth, "wq": self.wq, "wk": self.wk, "wv": self.wv,
                  "dense": self.dense, "dot_attention": self.dot_attention}
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, **kwargs):
        q, k, v = inputs
        batch_size = tf.shape(q)[0]

        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        # 分头
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # 通过缩放点积注意力层
        scaled_attention = self.dot_attention(q, k, v)  # (batch_size, num_heads, seq_len_q, depth)

        # “多头维度” 后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        # 合并 “多头维度”
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # 全连接层
        output = self.dense(concat_attention)

        return output


# dot attention
def scaled_dot_product_attention(q, k, v):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dim_k = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dim_k)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output


# 层标准化
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.eps = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"eps": self.eps}
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=tf.ones_initializer(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=tf.zeros_initializer(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=-1, keepdims=True)
        std = tf.keras.backend.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


# 前向网络
def point_wise_feed_forward_network(d_model, middle_units):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(middle_units, activation='relu'),
        tf.keras.layers.Dense(d_model, activation='relu')])