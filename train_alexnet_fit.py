import tensorflow as tf
import numpy as np
import time
import os

from src import alexnet_fit

num_steps = 10000
step = 50
useCkpt = False
# 定义一个存储路径
checkpoint_dir = os.getcwd() + '\\models_alexnet_fit\\'
keep_prob = tf.placeholder(tf.float32)


def inputs(filename, batch_size):
    image, label = alexnet_fit.read_file(filename)
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=batch_size,
                                            capacity=30000 + batch_size,
                                            min_after_dequeue=30000)
    return images, labels

# 构建模型
logits = alexnet_fit.alex_net(alexnet_fit.X, alexnet_fit.weights, alexnet_fit.biases, keep_prob)
prediction = tf.nn.softmax(logits)
# 定义损失函数和优化器
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=alexnet_fit.Y))
optimizer = tf.train.AdamOptimizer(learning_rate=alexnet_fit.learning_rate)
train_op = optimizer.minimize(loss=loss_op)
# 评估函数
correct_pred = tf.equal(tf.argmax(prediction, 1), alexnet_fit.Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#训练模型
init = tf.global_variables_initializer()
def train_model():
    global learning_rate
    time1 = time.time()
    for i in range(1, num_steps + 1):
        with tf.Graph().as_default():

            batch_x, batch_y = sess.run([images, labels])
            batch_x = np.reshape(batch_x, [alexnet_fit.batch_size, alexnet_fit.input_size])

            sess.run(train_op, feed_dict={alexnet_fit.X: batch_x, alexnet_fit.Y: batch_y, keep_prob: alexnet_fit.dropout})

            if i % step == 0 or i == 1:
                loss, acc = sess.run([loss_op, accuracy], feed_dict={alexnet_fit.X: batch_x, alexnet_fit.Y: batch_y, keep_prob: 1})
                learning_rate = update_learning_rate(acc, learn_rate=alexnet_fit.initial_learning_rate)
                # 存储模型
                saver.save(sess, checkpoint_dir + 'model.ckpt')
                time2 = time.time()
                print("time: %.4f step: %d loss: %.4f accuracy: %.4f" % (time2 - time1, i, loss, acc))
                time1 = time.time()


def update_learning_rate(acc, learn_rate):
    return learn_rate - acc * learn_rate * 0.9

# 声明完所有变量后，调用tf.train.Saver(),位于它之后的变量将不会被存储
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    tfrecords_name = 'D:\\Robots\\fruit_network\\train-00000-of-00001'
    images, labels = inputs(tfrecords_name, alexnet_fit.batch_size)
    #开始输入入队线程
    # 协调器，用来管理线程，协调线程间的关系可以视为一种信号量，用来做同步。
    coord = tf.train.Coordinator()
    #创建队列管理器
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if useCkpt:
        # 加载模型
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

    train_model()

    coord.request_stop()# 通知其他线程关闭
    coord.join(threads)# 等待其他线程结束，其他所有线程关闭之后，这一函数才能返回。
    sess.close()
