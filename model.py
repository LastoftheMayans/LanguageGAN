import tensorflow as tf
layers = tf.layers
import numpy as np
import os, sys
from corpus import Corpus

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# USAGE:
# model.py <TRAIN> <LOAD>
# if train == 0, then the model will test, else train
# if load == 0, then the model will not load, else load

BATCH_SIZE = 100
EPOCHS_PER_LENGTH_INCREASE = 1
NUM_GEN_UPDATES = 2

SENTENCE_SIZE = 12
NOISE_DIMENSION = 64
EMBEDDING_SIZE = 128

LEARNING_RATE = 0.001
BETA1 = 0.5

ITERS_PER_PRINT = 100
ITERS_PER_SAVE = 100
OUTFILE = "output.txt"
AUTHOR = "hamilton"

# Numerically stable logarithm function
def log(x):
    return tf.log(tf.maximum(x, 1e-5))

class Model:

    # @params
    # text_batch    tensor of shape [batch_size, sentence_size] for the real text input
    # g_input_z     tensor of shape [batch_size, sentence_size, noise_dimension] for the fake text seed
    # sentence_mask tensor of shake [sentence_size] with leading 1s and trailing 0s to mask off later words from earlier epochs
    # vocab_size    int representing the real size of the corpus vocabulary

    def __init__(self, text_batch, g_input_z, sentence_mask, vocab_size):
        self.text_batch = text_batch
        self.g_input_z = g_input_z
        self.sentence_mask = sentence_mask
        self.vocab_size = vocab_size

        self.g_output = self.generator()
        self.g_params = tf.trainable_variables()
        
        with tf.variable_scope("discriminator"):
            self.embedding = tf.Variable(tf.truncated_normal((self.vocab_size, EMBEDDING_SIZE), stddev=0.1, dtype=tf.float32))
            txt = tf.nn.embedding_lookup(self.embedding, self.text_batch)
            self.d_real = self.discriminator(txt)
        with tf.variable_scope("discriminator", reuse=True):
            self.d_fake = self.discriminator(self.g_output)
        self.d_params = [ v for v in tf.trainable_variables() if v.name.startswith("discriminator") ]

        self.output = self.embedding_lookup()

        self.g_loss = self.g_loss_function()
        self.d_loss = self.d_loss_function()
        self.g_train = self.g_trainer()
        self.d_train = self.d_trainer()
        self.evaluate = self.eval_function()

    def generator(self):
        rnnsize = 256

        with tf.variable_scope("generator"):
            rnn_cell = tf.contrib.rnn.GRUCell(rnnsize)
            outputs, state = tf.nn.dynamic_rnn(rnn_cell, self.g_input_z, dtype=tf.float32)

            g1 = layers.dense(outputs, EMBEDDING_SIZE)
            return g1

    def discriminator(self, embedding):
        # embedding is a tensor of shape [batch_size, sentence_size, embedding_size]
        rnnsize = 256

        rnn_cell = tf.contrib.rnn.GRUCell(rnnsize)
        outputs, state = tf.nn.dynamic_rnn(rnn_cell, embedding, dtype=tf.float32)

        d1 = layers.dense(outputs, 1)
        d2 = tf.reshape(d1, [-1, SENTENCE_SIZE])

        # mask off later words
        d3 = tf.multiply(d2, self.sentence_mask)

        d4 = layers.dense(d3, 1, activation=tf.nn.sigmoid)
        d5 = tf.reshape(d4, [-1])

        return d5

    # Training loss for Generator
    def g_loss_function(self):
        g_loss = tf.reduce_mean(-log(self.d_fake))
        return g_loss

    # Training loss for Discriminator
    def d_loss_function(self):
        d_loss = 0.5 * tf.reduce_mean((-log(self.d_real) - log(1 - self.d_fake)))
        return d_loss

    # Optimizer/Trainer for Generator
    def g_trainer(self):
        g_train = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1).minimize(self.g_loss, var_list = self.g_params)
        return g_train

    # Optimizer/Trainer for Discriminator
    def d_trainer(self):
        d_train = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1).minimize(self.d_loss, var_list = self.d_params)
        return d_train

    # Lookup closest english words from generator output
    def embedding_lookup(self):
        # cosine similarity, adapted from a helpful post on stackoverflow
        batch = tf.reshape(self.g_output, [-1, EMBEDDING_SIZE])
        normalized_embedding = tf.nn.l2_normalize(self.embedding, axis=1)
        normalized_batch = tf.nn.l2_normalize(batch, axis=1)
        cosine = tf.matmul(normalized_batch, tf.transpose(normalized_embedding, [1, 0]))
        words = tf.argmax(cosine, 1)
        return tf.reshape(words, [-1, SENTENCE_SIZE])

    # For evaluating the quality of generated text
    def eval_function(self):
        return tf.Variable(tf.truncated_normal([1], stddev=0.1))

## --------------------------------------------------------------------------------------

# Build model
g_input_z = tf.placeholder(tf.float32, (None, SENTENCE_SIZE, NOISE_DIMENSION))
txt_input = tf.placeholder(tf.int32, (None, SENTENCE_SIZE))
sentence_mask = tf.placeholder(tf.float32, (SENTENCE_SIZE))
corpus = Corpus(AUTHOR, batch_size=BATCH_SIZE, sentence_length=SENTENCE_SIZE)
model = Model(txt_input, g_input_z, sentence_mask, corpus.vocab_size)
num_batches = len(corpus) // BATCH_SIZE

# Start session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# For saving/loading models
saver = tf.train.Saver()

## --------------------------------------------------------------------------------------

# Load the last saved checkpoint during training or used by test
def load_last_checkpoint():
    saver.restore(sess, tf.train.latest_checkpoint('./'))

def gen_noise():
    return 2 * np.random.rand(BATCH_SIZE, SENTENCE_SIZE, NOISE_DIMENSION) - 1

def gen_mask(l):
    mask = [ 1 if i < l else 0 for i in range(SENTENCE_SIZE) ]
    return np.array(mask)

def make_gen_dict(l):
    return {g_input_z:gen_noise(), sentence_mask:gen_mask(l)}

def make_dis_dict(l):
    return {g_input_z:gen_noise(), txt_input: corpus.next_batch(), sentence_mask:gen_mask(l)}

# Train the model
num_epochs = EPOCHS_PER_LENGTH_INCREASE * SENTENCE_SIZE
def train():
    # Training loop
    for epoch in range(num_epochs):
        print('========================== EPOCH %d  ==========================' % epoch)

        mask_size = (epoch // EPOCHS_PER_LENGTH_INCREASE) + 1
        # Loop over our data until we run out
        for i in range(num_batches):
            loss_d, _ = sess.run([model.d_loss, model.d_train], feed_dict=make_dis_dict(mask_size))

            for j in range(NUM_GEN_UPDATES):
                loss_g, _ = sess.run([model.g_loss, model.g_train], feed_dict=make_gen_dict(mask_size))

            # Print losses
            if i % ITERS_PER_PRINT == 0 and i != 0:
                print('Iteration %d: Generator loss = %g | Discriminator loss = %g' % (i, loss_g, loss_d))
            # Save
            if i % ITERS_PER_SAVE == 0 and i != 0:
                saver.save(sess, './cache/gan_saved_model')

        # Save at the end of the epoch, too
        saver.save(sess, './cache/gan_saved_model')

        # out = sess.run(model.evaluate, feed_dict = {g_input_z: gen_noise()})
        # print('**** EVALUATION: %g ****' % out)


# Test the model by generating some samples from random latent codes
def test():
    output = sess.run(model.output, feed_dict=make_gen_dict(SENTENCE_SIZE))
    output = [ corpus.translate(l.tolist()) for l in output ]

    with open(OUTFILE, 'w') as f:
        f.write("\n".join(output))
        f.write("\n")

## --------------------------------------------------------------------------------------

if __name__ == '__main__':
    if(len(sys.argv) < 3):
        print("Usage: " + sys.argv[0] + " <train if 0> + <load if 0>")
        sys.exit(1)
    do_train = int(sys.argv[1]) == 0
    load = int(sys.argv[2]) == 0

    if load:
        load_last_checkpoint()

    if do_train:
        train()
    elif load:
        test()
    else:
        train()
        test()

