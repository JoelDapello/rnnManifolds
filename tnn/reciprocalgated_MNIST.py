import os, time, glob, argparse

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tnn import main
from tnn.reciprocalgaternn import tnn_ReciprocalGateCell
from tnn.convrnn import tnn_ConvBasicCell

'''This is an example of passing a custom cell to your model,
in this case a vanilla convRNN implemented from scratch,
which can serve as a template for more complex custom cells'''

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--gpus', default=['1'], nargs='*')
parser.add_argument('--ntimes', default=5, type=int)
parser.add_argument('--nsteps', default=int(1e5), type=lambda x: int(float(x)))
FLAGS, _ = parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(FLAGS.gpus)

batch_size = FLAGS.batch_size
NUM_TIMESTEPS = 4 # number of timesteps we are predicting on
NETWORK_DEPTH = 5 # number of total layers in our network
DATA_PATH = 'datasets/' # path where MNIST data will be automatically downloaded to

# we always unroll num_timesteps after the first output of the model
TOTAL_TIMESTEPS = NETWORK_DEPTH + NUM_TIMESTEPS 

# we unroll at least NETWORK_DEPTH times (3 in this case) so that the input can reach the output of the network
# note tau is the value of the memory decay (by default 0) at the readout layer and trainable_flag is whether the memory decay is trainable, which by default is False

BASE_NAME = '../json/5L_mnist28_recip345sig_noBN'
# BASE_NAME = '../json/5L_mnist28_recip345sig'
#BASE_NAME = '../json/VanillaRNN'

def model_func(input_images, ntimes=TOTAL_TIMESTEPS, 
    batch_size=batch_size, edges_arr=[], 
    base_name=BASE_NAME, 
    tau=0.0, trainable_flag=False):

    with tf.variable_scope("my_model"):
        # reshape the 784 dimension MNIST digits to be 28x28 images
        input_images = tf.reshape(input_images, [-1, 28, 28, 1])
        base_name += '.json'
        print('Using base: ', base_name)
        # creates the feedforward network graph from json
        G = main.graph_from_json(base_name)

        for node, attr in G.nodes(data=True):
            memory_func, memory_param = attr['kwargs']['memory']
            if 'cell_depth' in memory_param:
            # if 'out_depth' in memory_param:
                # this is where you add your custom cell
                # attr['cell'] = tnn_ConvBasicCell
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

def train(restore=True):

    # get MNIST images
    mnist = input_data.read_data_sets(DATA_PATH, one_hot=False)
    
    # create the model
    x = tf.placeholder(tf.float32, [batch_size, 784])
    
    y_ = tf.placeholder(tf.int64, [batch_size]) # predicting 10 outputs
    
    outputs = model_func(x, ntimes=TOTAL_TIMESTEPS, 
        batch_size=batch_size, edges_arr=[], 
        base_name=BASE_NAME, tau=0.0, trainable_flag=False)
    
    # setup the loss (average across time, the cross entropy loss at each timepoint 
    # between model predictions and labels)
    with tf.name_scope('cumulative_loss'):
        outputs_arr = [tf.squeeze(outputs[i]) for i in range(len(outputs))]
        cumm_loss = tf.add_n([tf.losses.sparse_softmax_cross_entropy(logits=outputs_arr[i], labels=y_) \
            for i in range(len(outputs))]) / len(outputs)
    
    # setup the optimizer
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cumm_loss)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #if restore:
        #    saver.restore(sess, save_path='./ckpts/model.ckpt')
        for i in range(20000):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            if i % 100 == 0:
                train_loss = cumm_loss.eval(feed_dict={x: batch_xs, y_: batch_ys})
                print('step %d, training loss %g' % (i, train_loss))
                saver.save(sess, './ckpts/model.ckpt', global_step=i)
            train_step.run(feed_dict={x: batch_xs, y_: batch_ys})


def get_features(ims, layer='imnetds'):
    # create the model
    x = tf.placeholder(tf.float32, [batch_size, 784])
    
    y_ = tf.placeholder(tf.int64, [batch_size]) # predicting 10 outputs
    
    outputs = model_func(x, ntimes=TOTAL_TIMESTEPS, 
        batch_size=batch_size, edges_arr=[], 
        base_name=BASE_NAME, tau=0.0, trainable_flag=False)
    
    # placeholder = tf.placeholder(shape=(None, ims[0].shape[0], ims[0].shape[1], 3), dtype=tf.float32)
    # basenet(placeholder, conv_only=True)
    
    op = tf.get_default_graph().get_operations()
    print('mark',[m.name for m in op if 'output' in m.name])
    target = tf.get_default_graph().get_tensor_by_name('my_model/{}_8/output:0'.format(layer))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, save_path='./ckpts/model.ckpt')

        n_batches = (len(ims) - 1) // batch_size + 1
        out = []
        for i in range(n_batches):
            batch = ims[batch_size * i: batch_size * (i + 1)]
            batch_out = sess.run(target, feed_dict={x: batch})
            out.append(batch_out)
        out = np.row_stack(out)
    return out

if __name__ == '__main__':
    #ims = np.random.random([batch_size,784])
    #out = get_features(ims)
    #print(out)
    #print(out.shape)
    train()
