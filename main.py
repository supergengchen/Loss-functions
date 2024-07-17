import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import sys

# 设置标志
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_source_dir', 'data/train', 'Training data directory')
tf.app.flags.DEFINE_string('test_source_dir', 'data/test', 'Testing data directory')
tf.app.flags.DEFINE_string('val_source_dir', 'data/val', 'Validation data directory')
tf.app.flags.DEFINE_string('train_sub', 'train_subjects.txt', 'Training subjects list')
tf.app.flags.DEFINE_string('test_sub', 'test_subjects.txt', 'Testing subjects list')
tf.app.flags.DEFINE_string('val_sub', 'val_subjects.txt', 'Validation subjects list')
tf.app.flags.DEFINE_string('mask_name', 'mask', 'Name of Mask file')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Batch size.')
tf.app.flags.DEFINE_float('L2Lambda', 5e-4, 'L2 lambda for regularization.')
tf.app.flags.DEFINE_boolean('use_bm', False, 'Whether to use batch normalization.')
tf.app.flags.DEFINE_boolean('first_stage', False, 'Whether it is first stage or not.')
tf.app.flags.DEFINE_string('loss_type', 'sml', 'Type of loss (l1, dl, sml).')
tf.app.flags.DEFINE_float('dl_lambda', 0.0, 'Lambda for discriminator loss.')
tf.app.flags.DEFINE_float('dl_lambda_mean', 1.0, 'Lambda for mean loss in sml.')
tf.app.flags.DEFINE_float('dl_lambda_var', 1.0, 'Lambda for variance loss in sml.')
tf.app.flags.DEFINE_float('decay_rate', 0.95, 'Decay rate for learning rate.')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'Decay steps for learning rate.')
tf.app.flags.DEFINE_float('gene_learn_rate', 0.0001, 'Learning rate for generator.')
tf.app.flags.DEFINE_float('bias', 0.0, 'Bias for initializing variables.')
tf.app.flags.DEFINE_float('beta1', 0.9, 'Beta1 for Adam optimizer.')
tf.app.flags.DEFINE_float('beta2', 0.999, 'Beta2 for Adam optimizer.')
tf.app.flags.DEFINE_integer('epoch', 200, 'Number of epochs.')
tf.app.flags.DEFINE_integer('eval_frequency', 10, 'Evaluation frequency.')
tf.app.flags.DEFINE_integer('sp_size_per_sub', 100, 'Sample size per subject.')
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoints', 'Checkpoint directory')
tf.app.flags.DEFINE_string('test_dir', 'test', 'Test directory')
tf.app.flags.DEFINE_string('data_type', 'dmri', 'Data type (e.g., dmri)')
tf.app.flags.DEFINE_integer('hidden_units', 256, 'Number of hidden units.')
tf.app.flags.DEFINE_integer('num_layers', 3, 'Number of layers in the model.')
tf.app.flags.DEFINE_integer('output_units', 3, 'Number of output units.')
tf.app.flags.DEFINE_integer('input_dim', 144, 'Input dimension.')
tf.app.flags.DEFINE_integer('output_dim', 3, 'Output dimension.')
tf.app.flags.DEFINE_integer('feature_dim', 3, 'Feature dimension.')
tf.app.flags.DEFINE_integer('source_dimension', 144, 'Source dimension.')
tf.app.flags.DEFINE_integer('target_dimension', 3, 'Target dimension.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for Adam optimizer.')

# 数据准备函数
def data_preparation(dir_name, subjects, sample_size):
    all_data = np.empty((0, sample_size, 144 + 3 + 3))  # 假设总维度为144（source） + 3（target） + 3（feature）
    for sub_idx in range(len(subjects)):
        print('Loading the data for subject {} (ID: {})'.format(sub_idx, subjects[sub_idx]))
        temp_data = np.load(os.path.join(dir_name, subjects[sub_idx], FLAGS.data_type + '.npy'))
        if sample_size == 0:
            sample_size = temp_data.shape[0]
        source_data = temp_data[0:sample_size, 0:144]
        target_data = temp_data[0:sample_size, 144:147]
        feature_data = temp_data[0:sample_size, 147:150]
        temp_all_data = np.hstack((source_data, target_data, feature_data))
        all_data = np.vstack((all_data, temp_all_data)) if all_data.size else temp_all_data
    return all_data

train_sub = [line.rstrip('\n') for line in open(FLAGS.train_sub)]
test_sub = [line.rstrip('\n') for line in open(FLAGS.test_sub)]
val_sub = [line.rstrip('\n') for line in open(FLAGS.val_sub)]

# 创建目录
if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.test_dir):
    os.makedirs(FLAGS.test_dir)

# 加载验证数据
print('Load validation data')
validation_data = data_preparation(FLAGS.val_source_dir, val_sub, FLAGS.sp_size_per_sub)

# 加载训练数据
print('Load training data')
training_data = data_preparation(FLAGS.train_source_dir, train_sub, FLAGS.sp_size_per_sub)

class Model:
    def __init__(self, sess, name, M, sd, td, fd):
        self.name = name
        self.sess = sess
        self.M, self.sd, self.td, self.fd = M, sd, td, fd
        self.regularization = FLAGS.L2Lambda
        self.regularizers = []
        self.build_graph(self.sd, self.td, self.fd)

    def build_graph(self, sd, td, fd):
        self.graph = self.sess.graph
        with self.graph.as_default():
            with tf.name_scope(self.name + '_inputs'):
                self.ph_data = tf.placeholder(tf.float32, (FLAGS.batch_size, sd), 'data')
                self.ph_target = tf.placeholder(tf.float32, (FLAGS.batch_size, td), 'target')
                self.ph_feature = tf.placeholder(tf.float32, (FLAGS.batch_size, fd), 'feature')
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')
                self.ph_phase_train = tf.placeholder(tf.bool, name='phase_train')

            x = self.ph_data
            with tf.variable_scope('generator'):
                for i, M in enumerate(self.M[0:-1]):
                    x = self.fc(x, M, self.ph_phase_train, 'g_fc{}'.format(i + 1), batch_norm=FLAGS.use_bm, relu=True)
                    x = tf.nn.dropout(x, self.ph_dropout)
                x = self.fc(x, self.M[-1], self.ph_phase_train, 'g_fc_out', batch_norm=False, relu=False)
            self.x_g = x

            if FLAGS.first_stage:
                x = self.ph_target
            with tf.variable_scope('discriminator'):
                M_loss = [150, 150, 150, fd]
                for i, M in enumerate(M_loss[0:-1]):
                    x = self.fc(x, M, self.ph_phase_train, 'd_fc{}'.format(i + 1), batch_norm=FLAGS.use_bm, relu=True)
                    x = tf.nn.dropout(x, self.ph_dropout)
                x = self.fc(x, M_loss[-1], self.ph_phase_train, 'd_fc_out', batch_norm=False, relu=False)
            self.x_d = x

            self.g_loss = tf.reduce_mean(tf.abs(self.x_g - self.ph_target))
            self.d_loss = tf.reduce_mean(tf.abs(self.x_d - self.ph_feature))

            if FLAGS.loss_type == 'l1':
                self.loss = self.g_loss
            elif FLAGS.loss_type == 'dl':
                self.loss = self.g_loss + FLAGS.dl_lambda * self.d_loss
            elif FLAGS.loss_type == 'sml':
                self.loss_mean_1 = tf.reduce_mean(
                    tf.abs(tf.reduce_mean(self.x_g[:, 0:9], axis=1) - tf.reduce_mean(self.ph_target[:, 0:9], axis=1)))
                self.loss_mean_2 = tf.reduce_mean(
                    tf.abs(tf.reduce_mean(self.x_g[:, 9:21], axis=1) - tf.reduce_mean(self.ph_target[:, 9:21], axis=1)))
                self.loss_mean_3 = tf.reduce_mean(tf.abs(
                    tf.reduce_mean(self.x_g[:, 21:38], axis=1) - tf.reduce_mean(self.ph_target[:, 21:38], axis=1)))
                self.loss_mean_4 = tf.reduce_mean(tf.abs(
                    tf.reduce_mean(self.x_g[:, 38:62], axis=1) - tf.reduce_mean(self.ph_target[:, 38:62], axis=1)))
                self.loss_mean_5 = tf.reduce_mean(tf.abs(
                    tf.reduce_mean(self.x_g[:, 62:96], axis=1) - tf.reduce_mean(self.ph_target[:, 62:96], axis=1)))
                self.loss_mean_6 = tf.reduce_mean(tf.abs(
                    tf.reduce_mean(self.x_g[:, 96:144], axis=1) - tf.reduce_mean(self.ph_target[:, 96:144], axis=1)))
                self.loss_var_1 = tf.reduce_mean(
                    tf.abs(self.reduce_var(self.x_g[:, 0:9], axis=1) - self.reduce_var(self.ph_target[:, 0:9], axis=1)))
                self.loss_var_2 = tf.reduce_mean(tf.abs(
                    self.reduce_var(self.x_g[:, 9:21], axis=1) - self.reduce_var(self.ph_target[:, 9:21], axis=1)))
                self.loss_var_3 = tf.reduce_mean(tf.abs(
                    self.reduce_var(self.x_g[:, 21:38], axis=1) - self.reduce_var(self.ph_target[:, 21:38], axis=1)))
                self.loss_var_4 = tf.reduce_mean(tf.abs(
                    self.reduce_var(self.x_g[:, 38:62], axis=1) - self.reduce_var(self.ph_target[:, 38:62], axis=1)))
                self.loss_var_5 = tf.reduce_mean(tf.abs(
                    self.reduce_var(self.x_g[:, 62:96], axis=1) - self.reduce_var(self.ph_target[:, 62:96], axis=1)))
                self.loss_var_6 = tf.reduce_mean(tf.abs(
                    self.reduce_var(self.x_g[:, 96:144], axis=1) - self.reduce_var(self.ph_target[:, 96:144], axis=1)))
                self.d_loss_mean = (
                                               self.loss_mean_1 + self.loss_mean_2 + self.loss_mean_3 + self.loss_mean_4 + self.loss_mean_5 + self.loss_mean_6) / 6
                self.d_loss_var = (
                                              self.loss_var_1 + self.loss_var_2 + self.loss_var_3 + self.loss_var_4 + self.loss_var_5 + self.loss_var_6) / 6
                self.d_loss = FLAGS.dl_lambda_mean * self.d_loss_mean + FLAGS.dl_lambda_var * self.d_loss_var
                self.loss = self.g_loss + self.d_loss
            else:
                print('Incorrect loss type. Please check.')
                sys.exit()

            tf.summary.scalar('loss/loss', self.loss)
            tf.summary.scalar('loss/g_loss', self.g_loss)
            tf.summary.scalar('loss/d_loss', self.d_loss)

            for regularizer in self.regularizers:
                self.loss += self.regularization * regularizer

            self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            self.opt = self.optimizer.minimize(self.loss, global_step=self.global_step)
            self.ema = tf.train.ExponentialMovingAverage(decay=0.995)
            self.maintain_averages_op = tf.group(self.ema.apply([self.loss]))

            self.summary = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep=100)

    def fc(self, x, M, phase_train, name, batch_norm=False, relu=True):
        input_dim = x.get_shape().as_list()[1]
        output_dim = M
        w = tf.get_variable(name + '_w', [input_dim, output_dim], tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable(name + '_b', [output_dim], initializer=tf.constant_initializer(0.0))
        tf.add_to_collection('losses', tf.nn.l2_loss(w))
        if batch_norm:
            fc = tf.contrib.layers.batch_norm(tf.matmul(x, w) + b, is_training=phase_train, scope=name + '_bn')
        else:
            fc = tf.matmul(x, w) + b
        if relu:
            fc = tf.nn.relu(fc)
        return fc

    def reduce_var(self, x, axis=None, keepdims=False):
        m = tf.reduce_mean(x, axis=axis, keepdims=True)
        devs_squared = tf.square(x - m)
        return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


def train_and_evaluate():
    with tf.Graph().as_default(), tf.Session() as sess:
        model = Model(sess, 'model', [FLAGS.hidden_units] * FLAGS.num_layers + [FLAGS.output_units],
                      FLAGS.input_dim, FLAGS.output_dim, FLAGS.feature_dim)
        sess.run(tf.global_variables_initializer())
        for epoch in range(FLAGS.epoch):
            # Shuffle and batch training data
            np.random.shuffle(training_data)
            num_batches = training_data.shape[0] // FLAGS.batch_size
            for i in range(num_batches):
                begin = i * FLAGS.batch_size
                data = training_data[begin:begin + FLAGS.batch_size, 0:144]
                target = training_data[begin:begin + FLAGS.batch_size, 144:147]
                feature = training_data[begin:begin + FLAGS.batch_size, 147:150]
                feed_dict = {model.ph_data: data, model.ph_target: target, model.ph_feature: feature,
                             model.ph_dropout: 0.5, model.ph_phase_train: True}
                _, loss, summary, step = sess.run([model.opt, model.loss, model.summary, model.global_step],
                                                  feed_dict=feed_dict)
                if (epoch + 1) % FLAGS.eval_frequency == 0:
                    print("Saving checkpoint...")
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model_epoch{}.ckpt'.format(epoch + 1))
                    model.saver.save(sess, checkpoint_path)

                    # Validation
                    print("Validation...")
                    start_time = datetime.now()
                    num_val_steps = validation_data.shape[0] // FLAGS.batch_size
                    loss_matrix = np.zeros((3, num_val_steps))

                    for val_step in range(num_val_steps):
                        val_begin = val_step * FLAGS.batch_size
                        BatchSource = validation_data[val_begin:val_begin + FLAGS.batch_size, 0:FLAGS.source_dimension]
                        BatchTarget = validation_data[val_begin:val_begin + FLAGS.batch_size,
                                      FLAGS.source_dimension:FLAGS.source_dimension + FLAGS.target_dimension]
                        BatchFeature = validation_data[val_begin:val_begin + FLAGS.batch_size,
                                       FLAGS.source_dimension + FLAGS.target_dimension:]

                        feed_dict = {
                            model.ph_data: BatchSource,
                            model.ph_target: BatchTarget,
                            model.ph_feature: BatchFeature,
                            model.ph_dropout: 1.0,
                            model.ph_phase_train: False
                        }

                        loss, g_loss, d_loss = sess.run([model.loss, model.g_loss, model.d_loss], feed_dict=feed_dict)
                        loss_matrix[:, val_step] = [loss, g_loss, d_loss]

                    # Calculate and print validation loss
                    loss_mean = np.mean(loss_matrix, axis=1)
                    print('Validation Loss: epoch {}, loss = {:.5}, g_loss = {:.5}, d_loss = {:.5}'.format(
                        epoch + 1, loss_mean[0], loss_mean[1], loss_mean[2]))

                    # Print validation time
                    print('Validation Time: {}'.format(datetime.now() - start_time))

        # Test
        test_and_evaluate(sess, model)


def test_and_evaluate(sess, model):
    print("Testing...")
    test_data = data_preparation(FLAGS.test_source_dir, test_sub, FLAGS.sp_size_per_sub)
    num_test_steps = test_data.shape[0] // FLAGS.batch_size
    test_loss_matrix = np.zeros((3, num_test_steps))

    for test_step in range(num_test_steps):
        test_begin = test_step * FLAGS.batch_size
        BatchSource = test_data[test_begin:test_begin + FLAGS.batch_size, 0:FLAGS.source_dimension]
        BatchTarget = test_data[test_begin:test_begin + FLAGS.batch_size,
                      FLAGS.source_dimension:FLAGS.source_dimension + FLAGS.target_dimension]
        BatchFeature = test_data[test_begin:test_begin + FLAGS.batch_size,
                       FLAGS.source_dimension + FLAGS.target_dimension:]

        feed_dict = {
            model.ph_data: BatchSource,
            model.ph_target: BatchTarget,
            model.ph_feature: BatchFeature,
            model.ph_dropout: 1.0,
            model.ph_phase_train: False
        }

        loss, g_loss, d_loss = sess.run([model.loss, model.g_loss, model.d_loss], feed_dict=feed_dict)
        test_loss_matrix[:, test_step] = [loss, g_loss, d_loss]

    # Calculate and print test loss
    test_loss_mean = np.mean(test_loss_matrix, axis=1)
    print('Test Loss: loss = {:.5}, g_loss = {:.5}, d_loss = {:.5}'.format(
        test_loss_mean[0], test_loss_mean[1], test_loss_mean[2]))


if __name__ == '__main__':
    tf.app.run(main=train_and_evaluate)