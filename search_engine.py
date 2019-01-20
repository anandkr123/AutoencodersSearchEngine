from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters,             # C H A N G E D #
learning_rate = 0.01
num_steps = 1000
batch_size = 100
n=4
image = []
encoding = []

display_step = 10
examples_to_show = 10


# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_hidden_1 = 256 # 1st layer num features
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
# Building the encoder
def encoder(x,y):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    mean=y*0.05
    var=y*0.03
    noise = np.random.normal(0.01, 0.005, num_hidden_1)
    layer_1=layer_1+noise
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

# Building the decoder
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
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    noise_mag=1
    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x,nse:noise_mag})

        noise_mag=noise_mag+1
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

# till here to check the encoder and decoder functioning only, as to how they perform based on the loss value


# Creating list of train images and its encoded representation

    for i in range(num_steps+1):
        # MNIST test set
        batch_x, _ = mnist.train.next_batch(n)
        g = sess.run(encoder_op, feed_dict={X: batch_x, nse: 0})
        for i in range(0, batch_x.shape[0]):
            image.append(batch_x[i])
            x = np.rint(g[i])
            encoding.append(x)

print("\nTotal number of encodings, done from image data set:" , len(encoding))
print("\nDimensions of first encoded image : ", encoding[1].shape)


matrixOfEncodings=np.ndarray(shape=(len(encoding), len(encoding[0])), dtype=int)
print("\nMatrix with only the encoded values of the image:", matrixOfEncodings.shape)


i = 0
for entry in encoding:

    matrixOfEncodings[i] = entry
    i = i+1


# reading a test_image, encoding it and searching with the help OF XOR in the encoded (matrix of encodings) representation

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    batch_x, _ = mnist.test.next_batch(n)
    #print(batch_x.shape)
    x = np.array(batch_x[0]).ravel()                              # reading the first image from the test_data
    x = x.reshape([28, 28])
    copied_test_image = x;
    g = sess.run(encoder_op, feed_dict={X: batch_x, nse: 0})
    # Original Image
    # x = np.array(batch_x[0]).ravel()
    # plt.imshow(x.reshape([28, 28]))
    #Hidden Representation
    hidden_rep = g[0]                                             # storing the first encoded image after encoder_op
    # print(g[0].shape)
    # print(matrixOfEncodings.shape)
    listOfScores = []
    count = 0
    print("\nScores to visualize how well the encoded image matches with the test image for search")
    for vector in matrixOfEncodings:
        # print(vector.shape)
        # print(hidden_rep.shape)
        # print((np.bitwise_xor(np.ma.make_mask(vector),np.ma.make_mask(hidden_rep))))
        x = (np.bitwise_xor(np.ma.make_mask(vector), np.ma.make_mask(hidden_rep)))
        listOfScores.append(128-np.count_nonzero(x))

print("\n", listOfScores)

                                           # ADDED #
max_value = max(listOfScores)
max_index = listOfScores.index(max_value)
print("The index", max_index, "and the max value", max_value)

# the corresponding matched image

matched_image = image[max_index]
fig = plt.figure()
plt.subplot(1, 2, 1)

plt.imshow(matched_image.reshape([28, 28]))

# the corresponding test image that we are searching for
plt.subplot(1, 2, 2)
plt.imshow(copied_test_image)
plt.show()