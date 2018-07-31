import os, time, glob, argparse

import numpy as np
import tensorflow as tf

from tnn import main
from tnn.reciprocalgaternn import tnn_ReciprocalGateCell

host = os.uname()[1]
if host.startswith('braintree'):
    DATA_PATH = '/braintree/data2/active/users/qbilius/datasets/imagenet2012_tf_256px'
elif host.startswith('node'):
    DATA_PATH = '/om/user/qbilius/imagenet2012_tf_256px'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--test_batch_size', default=128, type=int)
parser.add_argument('--gpus', default=['1'], nargs='*')
parser.add_argument('--ntimes', default=5, type=int)
parser.add_argument('--nsteps', default=int(1e5), type=lambda x: int(float(x)))
FLAGS, _ = parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(FLAGS.gpus)
print(os.environ)

batch_size = FLAGS.batch_size
NUM_TIMESTEPS = 4 # number of timesteps we are predicting on
NETWORK_DEPTH = 5 # number of total layers in our network

# we always unroll num_timesteps after the first output of the model
TOTAL_TIMESTEPS = NETWORK_DEPTH + NUM_TIMESTEPS 
BASE_NAME = '../json/5L_imnet128_recip345sig_noBN'

def model_func(input_images, ntimes=TOTAL_TIMESTEPS, 
    batch_size=batch_size, edges_arr=[], 
    base_name=BASE_NAME, 
    tau=0.0, train=False, trainable_flag=False):

    with tf.variable_scope("my_model"):
        # I think images should show up already formatted
        # input_images = tf.reshape(input_images, [-1, 28, 28, 1])
        base_name += '.json'
        print('Using base: ', base_name)
        # creates the feedforward network graph from json
        G = main.graph_from_json(base_name)

        for node, attr in G.nodes(data=True):
            memory_func, memory_param = attr['kwargs']['memory']
            if 'cell_depth' in memory_param:
                # this is where you add your custom cell
                attr['cell'] = tnn_ReciprocalGateCell
            else:
                # default to not having a memory cell
                # tau = 0.0, trainable = False
                attr['kwargs']['memory'][1]['memory_decay'] = tau
                attr['kwargs']['memory'][1]['trainable'] = trainable_flag

        # add any non feedforward connections here: e.g. [('L2', 'L1')]
        G.add_edges_from(edges_arr)

        # initialize network to infer the shapes of all the parameters
        main.init_nodes(G, input_nodes=['L1'], batch_size=batch_size)
        # unroll the network through time
        main.unroll(G, input_seq={'L1': input_images}, ntimes=ntimes)

        outputs = {}
        # start from the final output of the model and 4 timesteps beyond that
        for t in range(ntimes-NUM_TIMESTEPS, ntimes):
            idx = t - (ntimes - NUM_TIMESTEPS) # keys start at timepoint 0
            outputs[idx] = G.node['readout']['outputs'][t]

        return outputs

class ConvRNNCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our GRU cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = 1

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state):
        conv_kwargs = dict(filters=self.filters,
                           kernel_size=self.kernel_size,
                           strides=self.strides,
                           padding='same',
                           activation=None,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                           #kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                           #bias_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                           reuse=tf.AUTO_REUSE)
        i = tf.layers.conv2d(inputs, name='input', **conv_kwargs)
        s = tf.layers.conv2d(state, name='state', **conv_kwargs)
        state = tf.contrib.layers.layer_norm(i + s,
                                             activation_fn=tf.nn.elu,
                                             reuse=tf.AUTO_REUSE,
                                             scope='layer_norm'
                                             )
        x = tf.layers.max_pooling2d(state, 3, 2, padding='same')
        output = tf.identity(x, name='output')
        return output, state

def basenet(inputs, train=False, conv_only=False):
    conv_kwargs = dict(padding='same',
                       activation=tf.nn.relu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                       #kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                       #bias_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                       reuse=tf.AUTO_REUSE)
    pool_kwargs = dict(padding='same')

    with tf.variable_scope('V1', reuse=tf.AUTO_REUSE):
        x = tf.layers.conv2d(inputs, 64, 7, strides=2, **conv_kwargs)
        x = tf.layers.max_pooling2d(x, 3, 2, **pool_kwargs)
        x = tf.identity(x, name='output')

    with tf.variable_scope('V2', reuse=tf.AUTO_REUSE):
        #x = tf.layers.conv2d(x, 128, 3, **conv_kwargs)
        x = tf.layers.conv2d(x, 256, 3, **conv_kwargs)
        x = tf.layers.max_pooling2d(x, 3, 2, **pool_kwargs)
        x = tf.identity(x, name='output')

    cells = []
    output = x
    print('output shape', output.shape.as_list())
    for i, layer in zip(range(3), ['V4', 'pIT', 'aIT']):
        with tf.variable_scope('rnn/multi_rnn_cell/cell_'.format(i)):
            #nfilters = 2 ** (8 + i)
            nfilters = 2 ** 8
            print("nfilters:", nfilters)
            state = tf.placeholder(shape=[None,
                                          output.shape.as_list()[1],
                                          output.shape.as_list()[2],
                                          nfilters,
                                          ],
                                   dtype=tf.float32)
            cell = ConvRNNCell(nfilters, 3)
            output, state = cell(output, state)
            cell._output_size = output.shape[1:]
            cell._state_size = state.shape[1:]
            cells.append(cell)

    cells = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, states = tf.nn.dynamic_rnn(cells, tf.stack([x] * FLAGS.ntimes, 0),
                                        dtype=tf.float32, time_major=True)
    x = tf.identity(outputs[-1], name='aIT/output')

    if not conv_only:
        with tf.variable_scope('ds'):
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, 1000,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                #kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                #bias_regularizer=tf.contrib.layers.l2_regularizer(5e-4),
                                reuse=tf.AUTO_REUSE)
            x = tf.identity(x, name='output')

    return x

def basenet2(inputs, train=False, conv_only=False):
    x = model_func(inputs, ntimes=TOTAL_TIMESTEPS,
        batch_size=batch_size, edges_arr=[],
        base_name=BASE_NAME, tau=0.0, trainable_flag=False)

    return x[3]

def parse_image(im):
    im = tf.decode_raw(im, np.uint8)
    im = tf.image.convert_image_dtype(im, dtype=tf.float32)
    im = tf.reshape(im, [256, 256, 3])
    return im


class Train(object):

    def __init__(self, arch, kind):
        self.kind = kind
        self.train = kind == 'train'
        self.name = self.kind

        targets = {}
        with tf.name_scope(self.kind):
            inputs = self.data()
            logits = arch(inputs['images'], train=self.train)
            print(logits)
            targets['softmax_loss'] = self.softmax_loss(inputs['labels'], logits)
            targets['loss'] = targets['softmax_loss']
            #targets['loss'] = targets['softmax_loss'] + self.reg_loss()
            if self.train:
                targets['learning_rate'] = self.learning_rate()
                self.optimizer = self.get_optimizer(targets['learning_rate'], targets['loss'])

            targets['top1'] = self.top_k(inputs['labels'], logits, 1)
            targets['top5'] = self.top_k(inputs['labels'], logits, 5)

        self.targets = targets


    def data(self, num_parallel_calls=5):
        filenames = glob.glob(os.path.join(DATA_PATH, '{}_*.tfrecords'.format(self.kind)))
        batch_size = FLAGS.batch_size if self.train else FLAGS.test_batch_size
        ds = tf.data.Dataset.from_tensor_slices(filenames)
        if self.train:
            ds = ds.shuffle(buffer_size=20)
        ds = ds.flat_map(tf.data.TFRecordDataset)
        ds = ds.map(self.parse_data, num_parallel_calls=num_parallel_calls)
        ds = ds.prefetch(batch_size)
        if self.train:
            ds = ds.shuffle(buffer_size=1250 + 2 * batch_size)
        ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        ds = ds.repeat(None if self.train else 1)
        self.iterator = ds.make_initializable_iterator()
        inputs = self.iterator.get_next()
        return inputs

    def parse_data(self, example_proto):
        feats = {'images': tf.FixedLenFeature((), tf.string),
                 'labels': tf.FixedLenFeature((), tf.int64),
                 'ids': tf.FixedLenFeature((), tf.string)}
        feats = tf.parse_single_example(example_proto, feats)
        im = parse_image(feats['images'])
        if self.train:
            im = tf.random_crop(im, size=(224, 224, 3))
            im.set_shape([224, 224, 3])  # otherwise fails in tf 1.4
            im = tf.image.random_flip_left_right(im)
            im = tf.image.resize_images(im, (128, 128))
        else:
            # im = tf.image.resize_images(im, (224, 224))
            im = tf.image.resize_images(im, (128, 128))

        feats['images'] = im
        return feats

    def softmax_loss(self, labels, logits):
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        softmax_loss = tf.reduce_mean(softmax_loss)
        return softmax_loss

    def reg_loss(self):
        return tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    def top_k(self, labels, logits, k=1):
        top = tf.nn.in_top_k(logits, labels, k)
        top = tf.reduce_mean(tf.cast(top, tf.float32))
        return top

    def learning_rate(self):
        learning_rate = tf.train.polynomial_decay(learning_rate=5e-3,
                                                  global_step=tf.train.get_global_step(),
                                                  decay_steps=FLAGS.nsteps,
                                                  end_learning_rate=5e-5)
        return learning_rate

    def get_optimizer(self, learning_rate, loss):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(learning_rate, .9, use_nesterov=True)
            # optimizer = tf.train.AdagradOptimizer(learning_rate)
            # optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss)

            capped_grads_and_vars = []
            for grad, var in grads_and_vars:
                if grad is not None:
                    # gradient clipping. Some gradients returned are 'None' because
                    # no relation between the variable and loss; so we skip those.
                    capped_grad = tf.clip_by_value(grad, -1., 1.)
                    # capped_grad, _ = tf.clip_by_global_norm(grad, -1., 1.)
                    capped_grads_and_vars.append((capped_grad, var))

            opt_op = optimizer.apply_gradients(capped_grads_and_vars,
                                               global_step=tf.train.get_global_step())
        return opt_op

    def __call__(self, sess):
        if self.train:
            start = time.time()
            rec, _ = sess.run([self.targets, self.optimizer])
            rec['dur'] = time.time() - start
            return rec
        else:
            results = {k:[] for k in self.targets}
            durs = []
            sess.run(self.iterator.initializer)
            while True:
                start = time.time()
                try:
                    res = sess.run(self.targets)
                except tf.errors.OutOfRangeError:
                    break
                durs.append(time.time() - start)
                for k, v in res.items():
                    results[k].append(v)

            rec = {k: np.mean(v) for k,v in results.items()}
            rec['dur'] = np.mean(durs)
        return rec


def train(restore=True,
          save_train_steps=50,
          save_val_steps=50,
          save_model_steps=50,
          ):

    tf.Variable(0, trainable=False, name='global_step')
    #train = Train(model_func, 'train')
    #val = [Train(model_func, 'val')]
    train = Train(basenet2, 'train')
    #val = [Train(basenet2, 'val')]

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #if restore:
        #    saver.restore(sess, save_path='model.ckpt-100000')
        sess.run(train.iterator.initializer)

        step = sess.run(tf.train.get_global_step())
        while step <= FLAGS.nsteps:
            step = sess.run(tf.train.get_global_step())
            results = {'step': step}

            #if step % save_val_steps == 0:
            #    for v in val:
            #        results[v.name] = v(sess)

            if step % save_model_steps == 0:
                saver.save(sess=sess,
                           save_path='./imnet.ckpt/model.ckpt',
                           global_step=tf.train.get_global_step())

            if step % save_train_steps == 0:
                results['train'] = train(sess)
            else:
                sess.run(train.optimizer)

            #if len(results) > 1:  # not only step is available
            print(results)


def get_features(ims, layer='aIT'):
    placeholder = tf.placeholder(shape=(None, ims[0].shape[0], ims[0].shape[1], 3), dtype=tf.float32)
    basenet(placeholder, conv_only=True)
    target = tf.get_default_graph().get_tensor_by_name('{}/output:0'.format(layer))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, save_path='model.ckpt-100000')

        n_batches = (len(ims) - 1) // FLAGS.test_batch_size + 1
        out = []
        for i in range(n_batches):
            batch = ims[FLAGS.test_batch_size * i: FLAGS.test_batch_size * (i + 1)]
            batch_out = sess.run(target, feed_dict={placeholder: batch})
            out.append(batch_out)
        out = np.row_stack(out)
    return out


if __name__ == '__main__':
    #ims = np.random.random([12,224,224,3])
    #get_features(ims)
    train()
