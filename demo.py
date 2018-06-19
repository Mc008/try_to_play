# -*- coding: utf-8 -*-
from PyQt4 import QtCore
from PyQt4 import QtGui
import sys
from design import Ui_Form
import tensorflow as tf
import numpy as np
import cv2
import alexnet_fit
from scipy.misc import imread, imresize
num_classes = 61
# Dict = {1: 'Apple Braeburn', 2: 'Apple Golden 1', 3: 'Apple Golden 2', 4: 'Apple Golden 3', 5: 'Apple Granny Smith', 6: 'Apple Red 1', 7: 'Apple Red 2', 8: 'Apple Red 3', 9: 'Apple Red Delicious', 10: 'Apple Red Yellow', 11: 'Apricot', 12: 'Avocado杏', 13: 'Avocado杏 ripe',14: 'Banana', 15: 'Banana Red', 16: 'Cactus仙人掌 fruit', 17:'Carambula', 18: ' Cherry樱桃',19:' Clementine',
#         20:' Cocos椰子',21:'Dates枣',22:'Granadilla百香果',23:'Grape Pink',24:'Grape White',25:' Grape White 2',26:'Grapefruit Pink',27: 'Grapefruit White',28:'Guava石榴',29:' Huckleberry橘类植物',30:' Kaki',31:'Kiwi猕猴桃',32:'Kumquats',33:' Lemon',34:' Lemon Meyer',35:' Limes',36:' Litchi荔枝',37:' Mandarine',38:' Mango',39:' Maracuja',40:' Nectarine油桃',41:' Orange ',42:'Papaya木瓜',
#         43:'Passion Fruit百香果',44:'Peach桃',45:' Peach桃 Flat',46:' Pear',47:' Pear Abate',48:' Pear Monster',49:' Pear Williams',50:' Pepino',51: 'Pineapple',52:' Pitahaya Red',53:'Plum 李子',54:' Pomegranate石榴',55:' Quince',56:' Raspberry什么玩意',57:' Salak', 58:'Strawberry草莓',59:' Tamarillo番茄',60:' Tangelo蜜柑与柚子'}
Dict = {1: '红苹果', 2: '黄苹果(软)', 3: '黄苹果(脆)', 4: '绿苹果', 5: '绿苹果(脆)', 6: '浅红苹果', 7: '黄红苹果', 8: '暗红苹果', 9: '蛇果', 10: '红黄苹果', 11: '杏', 12: '牛油果', 13: '熟牛油果',14: '香蕉', 15: '红香蕉', 16: '仙人掌果', 17:'杨桃', 18: '樱桃',19:'小柑橘',
        20:' 椰子',21:'枣',22:'百香果',23:'红葡萄',24:'白葡萄',25:'绿葡萄',26:'葡萄柚',27: '葡萄柚白',28:'番石榴',29:' 美洲越橘',30:'柿子',31:'猕猴桃',32:'金橘',33:'柠檬',34:'柠檬麦耶 ',35:'酸橙',36:' 荔枝',37:' 中国柑橘',38:'绿芒果',39:'鸡蛋果',40:'油桃',41:'橘子',42:'番木瓜',
         43:'爱情果',44:'桃',45:'扁桃',46:'梨',47:'软梨',48:'黄梨怪 ',49:'白梨',50:'人参果',51: '菠萝',52:'火龙果',53:'李子',54:'石榴',55:'黄榅桲',56:'紫红木莓',57:'蛇皮果', 58:'草莓',59:'小西红柿',60:'橘柚'}


checkpoint_dir ="F:\\Fruit_Recognition\\model\\model.ckpt"
def conv2d(name, x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)

def maxpool2d(name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

#规范化层
def norm(name, l_input, l_size=4):
    return tf.nn.lrn(l_input, l_size, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)



class mywindow(QtGui.QMainWindow, Ui_Form):

    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        #png_2 = QtGui.QPixmap("X:\\pywin\\fruit_image\\0.png").scaled(self.label_2.width(), self.label_2.height())
        #self.label_2.setPixmap(png_2)
        self.label.mousePressEvent = self.my_clicked
        self.img = 0
        self.paiimg = 0
        self.dir_str = "F:\\Fruit_Recognition\\photo2.png"
    def my_clicked(self, e):
        print('预测')

    def openimage(self):
   # 打开文件路径
   #设置文件扩展名过滤,注意用双分号间隔
        imgName= QtGui.QFileDialog.getOpenFileName(self,"打开图片")
        #利用qlabel显示图片
        png_1 = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(png_1)
        self.img = imgName

    def showimage(self):
   # 打开文件路径
   #设置文件扩展名过滤,注意用双分号间隔
        #利用qlabel显示图片


        cap = cv2.VideoCapture(0)
        while (1):
            # get a frame
            ret, frame = cap.read()
            # show a frame
            cv2.imshow("capture", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite(self.dir_str, frame)
                break
        png_2 = QtGui.QPixmap(self.dir_str).scaled(self.label_2.width(), self.label_2.height())
        self.label_2.setPixmap(png_2)
        cap.release()
        cv2.destroyAllWindows()

    def perdictimage(self):

        X = tf.placeholder('float', [1, 100, 100, 3])
        logits = alexnet_fit.alex_net(X, alexnet_fit.weights, alexnet_fit.biases, 1)
        prediction = tf.nn.softmax(logits)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(init)
        # im = cv2.imread(self.img)
        im = imread(self.img, mode='RGB')
        # cv2.imshow('a', im)
        # cv2.waitKey(0)
        # im2 = cv2.resize(im, (100, 100))
        im2 = imresize(im, (100, 100))
        saver.restore(sess, checkpoint_dir)
        output = sess.run(prediction, feed_dict={X: [im2]})
        # print(output)
        index = np.argmax(output[0])
        #print('the predict is:', index)
        # preds = (np.argsort(prediction)[::-1])[0:5]
        # for p in preds:
        #     print(image_classes[p], prediction[p])
        self.label_3.setText("所属类为: " +str(index)+"   名称为: "+ Dict[index])
        sess.close()



    def predictimage2(self):

        X = tf.placeholder('float', [1, 100, 100, 3])
        logits = alexnet_fit.alex_net(X, alexnet_fit.weights, alexnet_fit.biases, 1)
        prediction = tf.nn.softmax(logits)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(init)
        # im = cv2.imread(self.img)
        im = imread(self.dir_str, mode='RGB')
        # cv2.imshow('a', im)
        # cv2.waitKey(0)
        # im2 = cv2.resize(im, (100, 100))
        im2 = imresize(im, (100, 100))
        saver.restore(sess, checkpoint_dir)
        output = sess.run(prediction, feed_dict={X: [im2]})
        # print(output)
        index = np.argmax(output[0])
        # print('the predict is:', index)
        # preds = (np.argsort(prediction)[::-1])[0:5]
        # for p in preds:
        #     print(image_classes[p], prediction[p])
        self.label_5.setText("所属类为: " + str(index) + "   名称为: " + Dict[index])
        sess.close()

        # with tf.Graph().as_default():
        #     # X = tf.placeholder('float', [None, 30000])
        #     # X = tf.reshape(X, shape=[-1, 100, 100, 3])
        #     X = tf.placeholder('float', [1, 100, 100, 3])
        #
        #     # saver = tf.train.Saver(tf.global_variables())
        #
        #
        #     weights = {
        #         'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
        #         'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        #         'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
        #         'wc4': tf.Variable(tf.random_normal([5, 5, 128, 256])),
        #         'wc5': tf.Variable(tf.random_normal([5, 5, 256, 512])),
        #         'wd1': tf.Variable(tf.random_normal([4 * 4 * 512, 4096])),
        #         'wd2': tf.Variable(tf.random_normal([4096, 4096])),
        #         'out': tf.Variable(tf.random_normal([4096, num_classes]))
        #     }
        #     biases = {
        #         'bc1': tf.Variable(tf.random_normal([32])),
        #         'bc2': tf.Variable(tf.random_normal([64])),
        #         'bc3': tf.Variable(tf.random_normal([128])),
        #         'bc4': tf.Variable(tf.random_normal([256])),
        #         'bc5': tf.Variable(tf.random_normal([512])),
        #         'bd1': tf.Variable(tf.random_normal([4096])),
        #         'bd2': tf.Variable(tf.random_normal([4096])),
        #         'out': tf.Variable(tf.random_normal([num_classes])),
        #     }
        #     saver = tf.train.Saver()
        #     conv1 = conv2d('conv1', X, weights['wc1'], biases['bc1'])
        #     pool1 = maxpool2d('pool1', conv1, k=2)
        #     norm1 = norm('norm1', pool1, l_size=4)
        #
        #     conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
        #     pool2 = maxpool2d('pool2', conv2, k=2)
        #     norm2 = norm('norm2', pool2, l_size=4)
        #
        #     conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
        #     pool3 = maxpool2d('pool3', conv3, k=2)
        #     norm3 = norm('norm3', pool3, l_size=4)
        #
        #     conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])
        #     pool4 = maxpool2d('pool4', conv4, k=2)
        #     norm4 = norm('norm4', pool4, l_size=4)
        #
        #     conv5 = conv2d('conv5', norm4, weights['wc5'], biases['bc5'])
        #     pool5 = maxpool2d('pool5', conv5, k=2)
        #     norm5 = norm('norm5', pool5, l_size=4)
        #
        #     fc1 = tf.reshape(norm5, shape=[-1, weights['wd1'].get_shape().as_list()[0]])
        #     fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        #     fc1 = tf.nn.relu(fc1)
        #     fc1 = tf.nn.dropout(fc1, 0.75)
        #
        #     fc2 = tf.reshape(fc1, shape=[-1, weights['wd2'].get_shape().as_list()[0]])
        #     fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
        #     fc2 = tf.nn.relu(fc2)
        #     fc2 = tf.nn.dropout(fc2, 0.75)
        #
        #     out = tf.add(tf.matmul(fc2, weights['out']), biases['out'], name='softmax')
        #     prediction = tf.nn.softmax(out)
        #
        #     im = imread(self.dir_str,'mode = RGB')
        #     # cv2.imshow('a', im)
        #     # cv2.waitKey(0)
        #     im3 = imresize(im, (100, 100))
        #     # x_img = np.reshape(im, [-1, 10000])
        #
        #     init = tf.global_variables_initializer()
        #
        #     sess = tf.Session()
        #     sess.run(init)
        #     saver.restore(sess, checkpoint_dir)
        #     output = sess.run(prediction, feed_dict={X: [im3]})
        #     index = np.argmax(output[0])
        #     # index = max(output[0])
        #
        #     #print('picture: ', '\n', output)
        #
        #     #self.label_5.setText("所属种类为："+str(index))
        #     self.label_5.setText("所属类为:" +str(index)+" 名称为:"+ Dict[index])
        #     sess.close()
if __name__ == "__main__":
 import sys
 app = QtGui.QApplication(sys.argv)
 window = mywindow()
 window.show()
 sys.exit(app.exec_())