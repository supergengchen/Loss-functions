import numpy as np
import tensorflow as tf


# Define necessary flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_source_path', 'data/data.npy', 'Training data directory')
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


# Model definition
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

            self.t_vars = tf.trainable_variables()
            self.d_vars = [var for var in self.t_vars if 'd_' in var.name]
            self.g_vars = [var for var in self.t_vars if 'g_' in var.name]

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.disc_learn_rate = FLAGS.gene_learn_rate
            self.gene_learn_rate = FLAGS.gene_learn_rate
            if FLAGS.decay_rate != 1:
                self.disc_learn_rate = tf.train.exponential_decay(self.disc_learn_rate, self.global_step,
                                                                  FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)
                self.gene_learn_rate = tf.train.exponential_decay(self.gene_learn_rate, self.global_step,
                                                                  FLAGS.decay_steps, FLAGS.decay_rate, staircase=True)

            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                if FLAGS.first_stage:
                    self.d_optim = tf.train.AdamOptimizer(self.disc_learn_rate, FLAGS.beta1,
                                                          beta2=FLAGS.beta2).minimize(self.d_loss, var_list=self.d_vars,
                                                                                      global_step=self.global_step)
                else:
                    self.g_optim = tf.train.AdamOptimizer(self.gene_learn_rate, FLAGS.beta1,
                                                          beta2=FLAGS.beta2).minimize(self.loss, var_list=self.g_vars,
                                                                                      global_step=self.global_step)

        self.op_summary = tf.summary.merge_all()
        num_o_models = FLAGS.epoch // FLAGS.eval_frequency
        self.op_saver = tf.train.Saver(max_to_keep=num_o_models)

    def fc(self, x, M, phase_train, name, batch_norm=False, relu=False):
        with tf.variable_scope(name):
            h = tf.get_variable('h', [x.get_shape()[1], M], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', [M], initializer=tf.constant_initializer(FLAGS.bias))
            x = tf.matmul(x, h) + b
            self.regularizers.append(tf.nn.l2_loss(h))
            self.regularizers.append(tf.nn.l2_loss(b))
            if batch_norm:
                x = self.batch_norm(x, phase_train)
            if relu:
                x = tf.nn.relu(x)
            return x

    def batch_norm(self, x, phase_train):
        return tf.cond(phase_train,
                       lambda: tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, is_training=True,
                                                            updates_collections=None),
                       lambda: tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, is_training=False,
                                                            updates_collections=None))

    def reduce_var(self, x, axis=None, keepdims=False):
        mean = tf.reduce_mean(x, axis=axis, keepdims=True)
        devs_squared = tf.square(x - mean)
        return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)


# Training and evaluation process
def train_and_evaluate():
    with tf.Session() as sess:
        sd, td, fd = 144, 3, 3
        model = Model(sess, "DMRI_Model", [128, 64, 32, td], sd, td, fd)
        sess.run(tf.global_variables_initializer())

        # Generate random DMRI data and ground truth for demonstration （ this is ample for six shells data）
        load_data = np.load(FLAGS.data_source_path).astype(np.float32)
        data = load_data[:, 0:144]
        target = load_data[:, 144:147]
        feature = load_data[:, 147:150]

        def fetch_batch_data(batch_size):
            idx = np.random.choice(len(data), batch_size)
            return data[idx], target[idx], feature[idx]

        num_batches = len(data) // FLAGS.batch_size
        for epoch in range(FLAGS.epoch):
            avg_loss = 0
            avg_g_loss = 0
            avg_d_loss = 0
            for i in range(num_batches):
                batch_data, batch_target, batch_feature = fetch_batch_data(FLAGS.batch_size)
                feed_dict = {
                    model.ph_data: batch_data,
                    model.ph_target: batch_target,
                    model.ph_feature: batch_feature,
                    model.ph_dropout: 0.5,
                    model.ph_phase_train: True
                }
                if FLAGS.first_stage:
                    _, loss, g_loss, d_loss, learning_rate = sess.run([model.d_optim, model.loss, model.g_loss, model.d_loss, model.disc_learn_rate], feed_dict=feed_dict)
                else:
                    _, loss, g_loss, d_loss, learning_rate = sess.run([model.g_optim, model.loss, model.g_loss, model.d_loss, model.disc_learn_rate], feed_dict=feed_dict)
                avg_loss += loss / num_batches
                avg_g_loss += g_loss / num_batches
                avg_d_loss += d_loss / num_batches
            if epoch % FLAGS.eval_frequency == 0:
                print("Epoch:", epoch, "Average Loss:", avg_loss, "Prediction Loss:", avg_g_loss, "Microstructure Loss:", avg_d_loss)
                model.op_saver.save(sess, './model.ckpt', global_step=model.global_step)


if __name__ == '__main__':
    train_and_evaluate()