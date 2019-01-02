import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('mnist_data/',one_hot=True)
import os

class Net:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input_x')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.w1 = tf.Variable(tf.truncated_normal(shape=[784, 512], stddev=tf.sqrt(1/512),dtype=tf.float32), name='weights')
        self.b1 = tf.Variable(tf.zeros(512), dtype=tf.float32,name='bias')
        self.w2 = tf.Variable(tf.truncated_normal(shape=[512, 256], stddev=tf.sqrt(1 / 256), dtype=tf.float32))
        self.b2 = tf.Variable(tf.zeros(256), dtype=tf.float32)
        self.w3 = tf.Variable(tf.truncated_normal(shape=[256, 128], stddev=tf.sqrt(1 / 128), dtype=tf.float32))
        self.b3 = tf.Variable(tf.zeros(128), dtype=tf.float32)
        self.wo = tf.Variable(tf.truncated_normal(shape=[128, 10], stddev=tf.sqrt(1 / 64), dtype=tf.float32))
        self.bo = tf.Variable(tf.zeros(10), dtype=tf.float32)
    def forward(self):
        with tf.name_scope('forward'):
            '# 输出值，对输入归一化'
            self.y1 = tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(self.x, self.w1) + self.b1))
            self.y2 = tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(self.y1, self.w2) + self.b2))
            self.y3 = tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(self.y2, self.w3) + self.b3))
            self.yo = tf.layers.batch_normalization(tf.matmul(self.y3, self.wo) + self.bo)
            tf.add_to_collection('net-output', self.yo)

    def backward(self):
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.yo))
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
    def summary(self):
        with tf.name_scope('weight'):
            tf.summary.histogram('_w', self.wo)
            tf.summary.scalar('_max', tf.reduce_max(self.wo))
            tf.summary.scalar('_min', tf.reduce_min(self.wo))
            tf.summary.scalar('_mean', tf.reduce_mean(self.wo))


if __name__ == '__main__':
    net = Net()
    net.forward()
    net.backward()
    net.summary()

    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('./logs', sess.graph)

        for epoch in range(500000):
            xs, ys = mnist.train.next_batch(100)
            summary, error, _ = sess.run([merged, net.loss, net.optimizer], feed_dict={net.x: xs, net.y: ys})
            writer.add_summary(summary, epoch)
            writer.flush()
            # writer.close()
            if epoch % 500 == 0:
                test_xs, test_ys = mnist.train.next_batch(100)
                test_y = sess.run(net.yo, feed_dict={net.x: test_xs})
                print('损失', error)
                print(np.mean(np.array(np.argmax(test_y, axis=1) == np.argmax(test_ys, axis=1), dtype=np.float32)))


        writer.close()
        if not os.path.exists('./mympl2'):
            os.makedirs('./mympl2')
        saver = tf.train.Saver()
        saver.save(sess, './mympl2/mympl_data.ckpt')


