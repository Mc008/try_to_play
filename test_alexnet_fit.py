import numpy
import tensorflow as tf
import numpy as np
import os

from src import alexnet_fit

checkpoint_dir = os.getcwd() + '\\models_alexnet_fit\\'
keep_prob = tf.placeholder(tf.float32)

image_number = 9673
total_images = 9673


def inputs(filename, batch_size):
    image, label = alexnet_fit.read_file(filename)
    images, labels = tf.train.batch([image, label],
                                    batch_size=batch_size,
                                    capacity=total_images + batch_size)
    return images, labels


def test_model():
    global image_number
    correct = 0
    while image_number > 0:
        batch_x, batch_y = sess.run([images, labels])
        batch_x = np.reshape(batch_x, [alexnet_fit.batch_size, alexnet_fit.input_size])
        acc = sess.run([correct_pred], feed_dict={alexnet_fit.X: batch_x, alexnet_fit.Y: batch_y, keep_prob: 1})
        image_number = image_number - alexnet_fit.batch_size
        correct = correct + numpy.sum(acc)
        print("Predicted %d out of %d; partial accuracy %.4f" % (correct, total_images - image_number, correct / (total_images - image_number)))
    print(correct/total_images)


logits = alexnet_fit.alex_net(alexnet_fit.X, alexnet_fit.weights, alexnet_fit.biases, keep_prob)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                        labels=alexnet_fit.Y))
optimizer = tf.train.AdamOptimizer(learning_rate=alexnet_fit.learning_rate)
train_op = optimizer.minimize(loss=loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), alexnet_fit.Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    tfrecords_name = 'D:\\Robots\\fruit_network\\validation-00000-of-00001'
    images, labels = inputs(tfrecords_name, alexnet_fit.batch_size)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    test_model()

    coord.request_stop()
    coord.join(threads)
    sess.close()


