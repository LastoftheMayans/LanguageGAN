import tensorflow as tf
layers = tf.layers
import numpy as np
from scipy.misc import imsave
import os, sys
from corpus import Corpus
import tensorflow.contrib.gan as gan

# USAGE:
# model.py <TRAIN> <LOAD>
# if train == 0, then the model will test, else train
# if load == 0, then the model will not load, else load


BATCH_SIZE = 100
SENTENCE_SIZE = 12
NOISE_DIMENSION = 64
EMBEDDING_SIZE = 128
EPOCHS = 10
NUM_GEN_UPDATES = 2
ITERS_PER_PRINT = 100
ITERS_PER_SAVE = 100
OUTFILE = "output.txt"
LEARNING_RATE = 0.001
BETA1 = 0.5

# Numerically stable logarithm function
def log(x):
    return tf.log(tf.maximum(x, 1e-5))

class Model:

    def __init__(self, text_batch, g_input_z, vocab_size, embedding_size=128):
        self.text_batch = text_batch
        self.g_input_z = g_input_z
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.g_output = self.generator()
        self.g_params = tf.trainable_variables()
        
        with tf.variable_scope("discriminator"):
            self.d_real = self.discriminator(self.text_batch)
        with tf.variable_scope("discriminator", reuse=True):
            self.d_fake = self.discriminator(self.g_output)
        self.d_params = [v for v in tf.trainable_variables() if v.name.startswith("discriminator")]

        # Declare losses, optimizers(trainers) and fid for evaluation
        self.g_loss = self.g_loss_function()
        self.d_loss = self.d_loss_function()
        self.g_train = self.g_trainer()
        self.d_train = self.d_trainer()
        self.evaluate = self.eval_function()

    def generator(self):
        sz = BATCH_SIZE * SENTENCE_SIZE * self.vocab_size
        with tf.variable_scope("generator"):
            g1 = layers.dense(self.g_input_z, sz)
            g2 = tf.reshape(g1, [-1, SENTENCE_SIZE, self.vocab_size])
            print(g2)
            return tf.argmax(g2, axis=2)

    def discriminator(self, txt):
        hidden_size = 256

        E = tf.Variable(tf.truncated_normal((self.vocab_size, self.embedding_size), stddev=0.1, dtype=tf.float32))
        txt = tf.nn.embedding_lookup(E, txt)

        d1 = layers.dense(txt, hidden_size)
        d2 = layers.batch_normalization(d1)
        d3 = tf.nn.leaky_relu(d2)

        d4 = layers.dense(d3, 1)
        d5 = layers.batch_normalization(d4)
        d6 = tf.nn.leaky_relu(d5)
        d7 = tf.reshape(d6, [-1, SENTENCE_SIZE])

        d8 = layers.dense(d7, 1, activation=tf.nn.sigmoid)
        d9 = tf.reshape(d8, [-1])

        return d9

    # Training loss for Generator
    def g_loss_function(self):
        g_loss = tf.reduce_mean(-log(self.d_fake))
        return g_loss

    # Training loss for Discriminator
    def d_loss_function(self):
        d_loss = .5 * tf.reduce_mean((-log(self.d_real) - log(1 - self.d_fake)))
        return d_loss

    # Optimizer/Trainer for Generator
    def g_trainer(self):
        g_train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1 = BETA1).minimize(self.g_loss, var_list = self.g_params)
        return g_train

    # Optimizer/Trainer for Discriminator
    def d_trainer(self):
        d_train = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, beta1 = BETA1).minimize(self.d_loss, var_list = self.d_params)
        return d_train

    # For evaluating the quality of generated text
    def eval_function(self):
        return tf.Variable(tf.truncated_normal(1, stddev=0.1))


## --------------------------------------------------------------------------------------

if __name__ == '__main__':
    g_input_z = tf.placeholder(tf.float32, (None, NOISE_DIMENSION))
    txt_input = tf.placeholder(tf.int32, (None, SENTENCE_SIZE))
    corpus = Corpus("hemingway", batch_size=BATCH_SIZE, sentence_length=SENTENCE_SIZE)
    model = Model(txt_input, g_input_z, corpus.vocab_size, embedding_size=EMBEDDING_SIZE)
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
    return np.random_uniform(-1, 1, (BATCH_SIZE, NOISE_DIMENSION))

# Train the model
def train():
    # Training loop
    for epoch in range(EPOCHS):
        print('========================== EPOCH %d  ==========================' % epoch)

        # Loop over our data until we run out
        for i in range(num_batches):
            loss_d, _ = sess.run([model.d_loss, m.d_train], feed_dict={g_input_z: gen_noise(), txt_input: corpus.next_batch()})

            for j in range(NUM_GEN_UPDATES):
                loss_g, _ = sess.run([model.g_loss, model.g_train], feed_dict={g_input_z: gen_noise()})

            # Print losses
            if i % ITERS_PER_PRINT == 0:
                print('Iteration %d: Generator loss = %g | Discriminator loss = %g' % (iteration, loss_g, loss_d))
            # Save
            if i % ITERS_PER_SAVE == 0:
                saver.save(sess, './gan_saved_model')

        # Save at the end of the epoch, too
        saver.save(sess, './gan_saved_model')

        out = sess.run(model.evaluate, feed_dict = {g_input_z: gen_noise()})  # Use sess.run to get the inception distance value defined above
        print('**** EVALUATION: %g ****' % out)


# Test the model by generating some samples from random latent codes
def test():
    output = sess.run(m.d_loss, feed_dict = {g_input_z: gen_noise()})
    output = output.eval() # turn to numpy
    output = [ corpus.translate(l.tolist()) for l in output]

    with(OUTFILE, 'w') as f:
        w.write("\n".join(output))

    # Convert to uint8
    gen_img_batch = gen_img_batch.astype(np.uint8)
    # Save images to disk
    for i in range(0, args.batch_size):
        img_i = gen_img_batch[i]
        s = args.out_dir+'/'+str(i)+'.png'
        imsave(s, img_i)

## --------------------------------------------------------------------------------------

if __name__ == '__main__':
    if(len(sys.argv) < 3):
        print("Usage: " + sys.argv[0] + " <test> + <load>")
        sys.exit(1)
    do_test = int(sys.argv[1]) == 0
    load = int(sys.argv[2]) == 1

    if load or test:
        load_last_checkpoint()

    if do_test:
        test()
    else:
        train()

