import tensorflow as tf
# 12年的方法AlexNet，优点是 在全连接层使用了Dropout，防止过拟合，使用非线性激活函数RELU，大数据训练，LRN规范化层的使用。
# 5个卷积组，2层全连接图像特征，1层全连接分类特征 共8个部分。8层
# 本网络应用于本数据，测试准确率为94.97%
batch_size = 50
input_size = 30000
num_classes = 61
dropout = 0.8

X = tf.placeholder(tf.float32, [None, input_size], name="X")
Y = tf.placeholder(tf.int64, [batch_size], name="Y")

# initial_learning_rate = 0.01
initial_learning_rate = 0.001
final_learning_rate = 0.0001

learning_rate = initial_learning_rate


# 构建模型
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 从文件中读取并解析一个样本
def read_file(filename):
    file_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),# 图片是string类型
            'label': tf.FixedLenFeature([], tf.int64),# 标记是int64类型
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.reshape(image, [100, 100, 3])
    label = tf.cast(features['label'], tf.int32)
    return image, label


def _variable_with_weight_decay(name, shape, initializer):
    return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)

def conv2d(name, x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)

def maxpool2d(name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

#规范化层
def norm(name, l_input, l_size=4):
    return tf.nn.lrn(l_input, l_size, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
    'wc4': tf.Variable(tf.random_normal([5, 5, 128, 256])),
    'wc5': tf.Variable(tf.random_normal([5, 5, 256, 512])),
    'wd1': tf.Variable(tf.random_normal([4 * 4 * 512, 4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 4096])),
    'out': tf.Variable(tf.random_normal([4096, num_classes]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bc4': tf.Variable(tf.random_normal([256])),
    'bc5': tf.Variable(tf.random_normal([512])),
    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([4096])),
    'out': tf.Variable(tf.random_normal([num_classes])),
}


def alex_net(X, weights, biases, dropout):
    X = tf.reshape(X, shape=[-1, 100, 100, 3])

    conv1 = conv2d('conv1', X, weights['wc1'], biases['bc1'])
    pool1 = maxpool2d('pool1', conv1, k=2)
    norm1 = norm('norm1', pool1, l_size=4)

    conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    pool2 = maxpool2d('pool2', conv2, k=2)
    norm2 = norm('norm2', pool2, l_size=4)

    conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    pool3 = maxpool2d('pool3', conv3, k=2)
    norm3 = norm('norm3', pool3, l_size=4)

    conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])
    pool4 = maxpool2d('pool4', conv4, k=2)
    norm4 = norm('norm4', pool4, l_size=4)

    conv5 = conv2d('conv5', norm4, weights['wc5'], biases['bc5'])
    pool5 = maxpool2d('pool5', conv5, k=2)
    norm5 = norm('norm5', pool5, l_size=4)



    fc1 = tf.reshape(norm5, shape=[-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    fc2 = tf.reshape(fc1, shape=[-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'], name='softmax')
    return out
