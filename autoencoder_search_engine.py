from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters,             # C H A N G E D #
learning_rate = 0.001
num_steps = 101
batch_size = 200
n=4
image = []
encoding = []

display_step = 10
examples_to_show = 10


# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_hidden_1 = 484 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)

X = tf.placeholder("float", [None, num_input])
nse = tf.placeholder("float", None)

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Encoding the image
# Building the encoder, the contraction of the image
def encoder(x,y):
    # Encoder Hidden layer with sigmoid activation #1
    noise = np.random.normal(0.5, 0.3, num_input)
    # adding noise to the inputs and clipping the value to range [0,1]
    x = x + noise
    x = tf.clip_by_value(x, 0., 1.)
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    mean = y*0.05
    var = y*0.03

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

# Building the decoder, reconstruction of the image
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


# Construct model
encoder_op = encoder(X, nse)

decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op

# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
# loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
# print(loss.shape)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Saving only recent last 2 checkpoints
    saver = tf.train.Saver(max_to_keep=2)

    # returns a CheckpointState if the state was available, None otherwise
    ckpt = tf.train.get_checkpoint_state('./')

    noise_mag = 1
    # Training

    if ckpt:
        print('Reading last checkpoint....')
        # saver = tf.train.import_meta_graph('autoencoder-model-3000.meta')

        # restoring the model from the last checkpoint
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        print('Model restored')
    else:
        print('Creating the checkpoint and models')


    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, nse: noise_mag})

        # noise_mag=noise_mag+1
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step  %i: Minibatch Loss: %f' % (i, l))
        if i % 99 == 0:
            val_batch_x, _ = mnist.test.next_batch(batch_size)

            # VALIDATION LOSS
            val_l = sess.run(loss, feed_dict={X: val_batch_x, nse: noise_mag})
            print('RECONSTRUCTION ERROR ON VALIDATION SET at  %i: : %f' % (i, val_l))

# till here to train our model, learn weights
        if i%1000 == 0:
            saver.save(sess, './autoencoder-model', global_step=i, write_meta_graph=False)


# Creating list of train images and its encoded representation

    for i in range(num_steps+1):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        g = sess.run(encoder_op, feed_dict={X: batch_x, nse: 0})
        for i in range(0, batch_x.shape[0]):
            image.append(batch_x[i])
            x = np.rint(g[i])
            encoding.append(x)

print("\nTotal number of encodings prepared from MNIST image test data set  {}, each encoding of length {}".format(len(encoding), encoding[1].shape))
#
matrixOfEncodings=np.ndarray(shape=(len(encoding), len(encoding[0])), dtype=int)
#

i = 0
for entry in encoding:
    matrixOfEncodings[i] = entry
    i = i+1

#
# reading a test_image, encoding it and searching with the help of XOR in the encoded (matrix of encodings) representation
#
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    batch_x, _ = mnist.test.next_batch(n)
    # print(batch_x.shape)
    x = np.array(batch_x[0]).ravel()                              # reading the first image from the test_data
    x = x.reshape([28, 28])
    copied_test_image = x;
    g = sess.run(encoder_op, feed_dict={X: batch_x, nse: 0})
    # Hidden Representation
    hidden_rep = g[0]                                             # storing the first encoded image after encoder_op
    listOfScores = []
    count = 0
    print("\nScores to visualize how well the encoded image matches with the test image for search")
    for vector in matrixOfEncodings:
        x = (np.bitwise_xor(np.ma.make_mask(vector), np.ma.make_mask(hidden_rep)))
        listOfScores.append(128-np.count_nonzero(x))

# print("\n", len(listOfScores))
def top_3_index(a):
    top_3_idx = np.argsort(a)[-3:]
    top_3_values = [a[i] for i in top_3_idx]
    return top_3_idx
                                           # ADDED #
max_value = max(listOfScores)
top_3_max = top_3_index(listOfScores)
print("The two top max index", top_3_max[1],top_3_max[2], "and the MAX value is", listOfScores[top_3_max[1]], listOfScores[top_3_max[2]])

# the corresponding matched image

matched_image_1= image[top_3_max[0]]
matched_image_2= image[top_3_max[1]]
matched_image_3= image[top_3_max[2]]
fig = plt.figure()

plt.subplot(2, 3, 1)
plt.imshow(matched_image_1.reshape([28, 28]))

plt.subplot(2, 3, 2)
plt.imshow(matched_image_2.reshape([28, 28]))
# The best matched image from the repository matrix
plt.subplot(2, 3, 3)
plt.imshow(matched_image_3.reshape([28, 28]))


# the corresponding test image that we are searching for
plt.subplot(2, 3, 5)
plt.imshow(copied_test_image)

plt.show()
