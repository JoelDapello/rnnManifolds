import os, time, glob, argparse

import numpy as np
from scipy.io import loadmat, savemat
import tensorflow as tf

from tnn import main
from tnn.reciprocalgaternn import tnn_ReciprocalGateCell

import warnings
warnings.filterwarnings("ignore")

host = os.uname()[1]
if host.startswith('braintree'):
    DATA_PATH = '/braintree/data2/active/users/qbilius/datasets/imagenet2012_tf_256px'
elif host.startswith('node'):
    DATA_PATH = '/om/user/qbilius/imagenet2012_tf_256px'

parser = argparse.ArgumentParser()
parser.add_argument('--train', default=0, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--test_batch_size', default=16, type=int)
parser.add_argument('--gpus', default=['1'], nargs='*')
parser.add_argument('--ntimes', default=5, type=int)
parser.add_argument('--nsteps', default=int(4e5), type=lambda x: int(float(x)))
FLAGS, _ = parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(FLAGS.gpus)

batch_size = FLAGS.batch_size
NUM_TIMESTEPS = 1  # number of timesteps we are predicting on
NETWORK_DEPTH = 16 # number of total layers in our network

# we always unroll num_timesteps after the first output of the model
TOTAL_TIMESTEPS = NETWORK_DEPTH + NUM_TIMESTEPS 
BASE_NAME = './json/resnet34_noBN'

def model_func(input_images, ntimes=TOTAL_TIMESTEPS, 
    batch_size=batch_size, edges_arr=[], 
    base_name=BASE_NAME, 
    tau=0.0, train=False, trainable_flag=False):

    # model_name = 'my_model'
    model_name = base_name.split('/')[-1]
    with tf.variable_scope(model_name, reuse=tf.AUTO_REUSE):
        base_name += '.json'
        print('Using model {} from {}'.format(model_name, base_name))
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
        main.init_nodes(G, input_nodes=['conv1'], batch_size=batch_size)
        # unroll the network through time
        main.unroll(G, input_seq={'conv1': input_images}, ntimes=ntimes)

        outputs = {}
        # start from the final output of the model and 4 timesteps beyond that
        for t in range(ntimes-NUM_TIMESTEPS, ntimes):
            idx = t - (ntimes - NUM_TIMESTEPS) # keys start at timepoint 0
            outputs[idx] = G.node['imnetds']['outputs'][t]

        return outputs


def basenet2(inputs, train=False, conv_only=False):
    x = model_func(inputs, ntimes=TOTAL_TIMESTEPS,
        batch_size=batch_size, edges_arr=[],
        base_name=BASE_NAME, tau=0.0, trainable_flag=False)

    return x[0]


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


def train(restore=False,
          save_train_steps=500,
          save_val_steps=5000,
          save_model_steps=1000,
          ):

    tf.Variable(0, trainable=False, name='global_step')
    train = Train(basenet2, 'train')
    val = [Train(basenet2, 'val')]
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if restore:
            saver.restore(sess, save_path='./resnet-imnet.ckpt/model.ckpt-test')
        sess.run(train.iterator.initializer)

        step = sess.run(tf.train.get_global_step())
        while step <= FLAGS.nsteps:
            step = sess.run(tf.train.get_global_step())
            results = {'step': step}

            if step % save_val_steps == save_val_steps-1:
                for v in val:
                    results[v.name] = v(sess)

            if step % save_model_steps == 0:
                saver.save(sess=sess,
                           save_path='./resnet-imnet.ckpt/model.ckpt',
                           global_step=tf.train.get_global_step())

            if step % save_train_steps == 0:
                results['train'] = train(sess)
            else:
                sess.run(train.optimizer)

            if len(results) > 1:  # not only step is available
                print(results)


def get_features(ims):
    n_batches = (len(ims) - 1) // FLAGS.test_batch_size + 1
    stack_depth = ims.shape[0]/n_batches
    placeholder = tf.placeholder(shape=(stack_depth, ims[0].shape[0], ims[0].shape[1], 3), dtype=tf.float32)
    # placeholder = tf.placeholder(shape=(None, ims[0].shape[0], ims[0].shape[1], 3), dtype=tf.float32)
    # ims = tf.tensor
    # placeholder = tf.placeholder(shape=tf.shape(ims), dtype=tf.float32)

    print('placeholder', placeholder)
    basenet2(placeholder, conv_only=True)
    
    ops = tf.get_default_graph().get_operations()
    layers = [op.name for op in ops if 'output' in op.name]
    print('target layers = ', layers)
    # target = tf.get_default_graph().get_tensor_by_name('{}/output:0'.format(layer))
    targets = [tf.get_default_graph().get_tensor_by_name(layer+':0') for layer in layers]

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, save_path='./resnet-imnet.ckpt/model.ckpt-test')

        n_batches = (len(ims) - 1) // FLAGS.test_batch_size + 1
        out = []
        for i in range(n_batches):
            print('running batch {} of {}'.format(i+1, n_batches))
            batch = ims[FLAGS.test_batch_size * i: FLAGS.test_batch_size * (i + 1)]
            batch_out = sess.run(targets, feed_dict={placeholder: batch})
            # batch_out = sess.run(target, feed_dict={placeholder: batch})
            out.append(batch_out)
        #out = np.row_stack(out)
    return layers, out


def load_HvM_images():
    HvM_file = loadmat('../imageData/HvM_128px.mat')
    imgs = HvM_file['imgs']
    return imgs


def save_HvM_features():
    # ims = np.random.random([128,128,128,3])
    # ims = load_HvM_images()[:256]
    ims = load_HvM_images()
    layers, out = get_features(ims)
    # print(out)
    #import code
    #code.interact(local=locals())
    import h5py

    out_idx = range(len(out[0]))
    print(out_idx)
    for i in out_idx: 
        features = np.row_stack([j[i] for j in out])
        print('saving features for layer:{}, shaped:{}'.format(layers[i],features.shape))
        savemat('../featureData/recipgated_HvM_{}_features.mat'.format(layers[i].replace('/','-')), {
            'features':features
        })

    print("saved HvM features!")


if __name__ == '__main__':
    if FLAGS.train:
        print('>>> TRAIN MODEL')
        train()
    else:
        print('>>> GET MODEL FEATURES')
        save_HvM_features()
