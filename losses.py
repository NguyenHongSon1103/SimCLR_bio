# ==============================================================================
# Code modified from NT-XENT-loss:
# https://github.com/google-research/simclr/blob/master/objective.py
# ==============================================================================
# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.activations import softmax
LARGE_NUM = 1e9

class SoftmaxCosineSim(keras.layers.Layer):
    """Custom Keras layer: takes all z-projections as input and calculates
    output matrix which needs to match to [I|O|I|O], where
            I = Unity matrix of size (batch_size x batch_size)
            O = Zero matrix of size (batch_size x batch_size)
    """

    def __init__(self, batch_size, feat_dim, temperature = 0.1, **kwargs):
        super(SoftmaxCosineSim, self).__init__()
        self.batch_size = batch_size
        self.feat_dim = feat_dim
        self.units = (batch_size, 4 * feat_dim)
        self.input_dim = [(None, feat_dim)] * (batch_size * 2)
        self.temperature = temperature
        self.LARGE_NUM = 1e9

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "batch_size": self.batch_size,
                "feat_dim": self.feat_dim,
                "units": self.units,
                "input_dim": self.input_dim,
                "temperature": self.temperature,
                "LARGE_NUM": self.LARGE_NUM,
            }
        )
        return config

    def call(self, inputs):
        z1 = []
        z2 = []

        for index in range(self.batch_size):
            # 0-index assumes that batch_size in generator is equal to 1
            z1.append(tf.math.l2_normalize(inputs[index][0], -1)) 
            z2.append(
                tf.math.l2_normalize(inputs[self.batch_size + index][0], -1)
            )
        z1 = tf.math.l2_normalize(inputs)
        # Gather hidden1/hidden2 across replicas and create local labels.
        z1_large = z1
        z2_large = z2

        masks = tf.one_hot(tf.range(self.batch_size), self.batch_size)

        # Products of vectors of same side of network (z_i), count as negative examples
        # Values on the diagonal are put equal to a very small value
        # -> exclude product between 2 identical values, no added value
        logits_aa = tf.matmul(z1, z1_large, transpose_b=True) / self.temperature
        logits_aa = logits_aa - masks * self.LARGE_NUM

        logits_bb = tf.matmul(z2, z2_large, transpose_b=True) / self.temperature
        logits_bb = logits_bb - masks * self.LARGE_NUM

        # Similarity between two transformation sides of the network (z_i and z_j)
        # -> diagonal should be as close as possible to 1
        logits_ab = tf.matmul(z1, z2_large, transpose_b=True) / self.temperature
        logits_ba = tf.matmul(z2, z1_large, transpose_b=True) / self.temperature

        part1 = softmax(tf.concat([logits_ab, logits_aa], 1))
        part2 = softmax(tf.concat([logits_ba, logits_bb], 1))
        output = tf.concat([part1, part2], 1)

        return output

def softmax_cosine_sim(inputs):
    def _loss(LARGE_NUM=1e9, temperature=0.1):
        z1 = []
        z2 = []
        bs = tf.shape(inputs)[0]
        for index in range(bs):
            # 0-index assumes that batch_size in generator is equal to 1
            z1.append(tf.math.l2_normalize(inputs[index][0], -1))
            z2.append(
                tf.math.l2_normalize(inputs[bs + index][0], -1)
            )
        z1 = tf.math.l2_normalize(inputs)
        # Gather hidden1/hidden2 across replicas and create local labels.
        z1_large = z1
        z2_large = z2

        masks = tf.one_hot(tf.range(bs), bs)

        # Products of vectors of same side of network (z_i), count as negative examples
        # Values on the diagonal are put equal to a very small value
        # -> exclude product between 2 identical values, no added value
        logits_aa = tf.matmul(z1, z1_large, transpose_b=True) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM

        logits_bb = tf.matmul(z2, z2_large, transpose_b=True) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM

        # Similarity between two transformation sides of the network (z_i and z_j)
        # -> diagonal should be as close as possible to 1
        logits_ab = tf.matmul(z1, z2_large, transpose_b=True) / temperature
        logits_ba = tf.matmul(z2, z1_large, transpose_b=True) / temperature

        part1 = softmax(tf.concat([logits_ab, logits_aa], 1))
        part2 = softmax(tf.concat([logits_ba, logits_bb], 1))
        output = tf.concat([part1, part2], 1)

        return output
    return _loss

def contrastive_loss(inputs, temperature=1.0):

    """Compute loss for model.
    Args:
        hidden: list 2 (`Tensor`), each of shape (bsz, dim).
        temperature: a `floating` number for temperature scaling.
    Returns:
        A loss scalar.
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
    """
    # Get (normalized) hidden1 and hidden2.
    output_a, output_b = inputs[0], inputs[1]
    output_a = tf.math.l2_normalize(output_a, -1)
    output_b = tf.math.l2_normalize(output_b, -1)

    batch_size = tf.shape(output_a)[0]

    output_a_large = output_a
    output_b_large = output_b
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(output_a, output_a_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(output_b, output_b_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(output_a, output_b_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(output_b, output_a_large, transpose_b=True) / temperature

    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ba, logits_bb], 1))
    loss = tf.reduce_mean(loss_a + loss_b)

    return loss #, logits_ab, labels