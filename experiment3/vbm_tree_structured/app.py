import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
import os
_ = tf.logging.set_verbosity(tf.logging.ERROR)

import variational_bayesian_memory as VBM
import data as data
import callbacks as calls
import routines as routines

flags = tf.app.flags


flags.DEFINE_enum(
    "mode", 'train', ['train', 'eval', 'generate', 'reconstruct', 'iterative_read', 'copy'],
    "mode: one of train, eval, generate, reconstruct, iterative_read, copy")

flags.DEFINE_enum("dataset", 'cifar10', ['mnist', 'omniglot', 'cifar10', 'celeb_a'], "dataset: which dataset to use")
flags.DEFINE_integer("img_height", 32, "img_height: height to scale images to, in pixels")
flags.DEFINE_integer("img_width", 32, "img_width: width to scale images to, in pixels")
flags.DEFINE_integer("img_channels", 3, "img_channels: number of image channels")
flags.DEFINE_boolean("discrete_outputs", False, "discrete_outputs: affects data preprocessing (e.g., grayscale->binarized), and determines a likelihood specification")

flags.DEFINE_integer("batch_size", 16, "batch_size: number of episodes per minibatch")
flags.DEFINE_integer("episode_len", 64, "episode_len: number of observations per episode")
flags.DEFINE_integer("num_hops", 2, "num_hops: number of memory hops")
flags.DEFINE_integer("num_clusters", 10, "num_clusters: number of memory clusters")
flags.DEFINE_integer("memory_size", 6, "memory_size: number of memory rows per cluster")
flags.DEFINE_integer("code_size", 200, "code_size: dimension of a code vector")
flags.DEFINE_integer("opt_iters", 10, "opt_iters: number of iterations to run writing algorithm; also used during iterative reading to optimize local latents q(y).")

flags.DEFINE_float("sr_alpha", 8.0, "sr_alpha: alpha hyperparameter for the stochastic regularization method introduced in section 4.3")
flags.DEFINE_float("sr_beta", 8.0, "sr_beta: beta hyperparameter for the stochastic regularization method introduced in section 4.3")
flags.DEFINE_float("sr_gamma", 0.50, "sr_gamma: gamma hyperparameter for the stochastic regularization method introduced in section 4.3")
flags.DEFINE_float("sr_delta", 0.20, "sr_delta: delta hyperparameter for the stochastic regularization method introduced in section 4.3")
flags.DEFINE_float("sr_epsilon", 0.10, "sr_epsilon: epsilon hyperparameter for the stochastic regularization method introduced in section 4.3")

flags.DEFINE_integer("enc_blocks", 3, "enc_blocks: number of blocks in the encoder")
flags.DEFINE_integer("dec_blocks", 3, "dec_blocks: number of blocks in the decoder")
flags.DEFINE_integer("num_filters", 64, "num_filters: number of convolutional filters per layer")

flags.DEFINE_float("lr", 0.0005, "lr: alpha parameter for Adam optimizer")

flags.DEFINE_string("summaries_dir", '/tmp/vbm_tree_structured_kmpp_swishgroupconvres_nsr_summaries/', "summaries_dir: directory for tensorboard logs")
flags.DEFINE_string("output_dir", 'output/', "output_dir: directory for visualizations")

flags.DEFINE_string("checkpoint_dir", 'checkpoints/', "checkpoint_dir: directory for saving model checkpoints")
flags.DEFINE_string("load_checkpoint", '', "load_checkpoint: checkpoint directory or checkpoint to load")

flags.DEFINE_integer("epochs", 1, "epochs: number of epochs to train for. ignored if mode is not 'train'")

FLAGS = flags.FLAGS


def main(_):

    ## hyperparams
    hps = tf.contrib.training.HParams(
        img_height = FLAGS.img_height,
        img_width = FLAGS.img_width,
        img_channels = FLAGS.img_channels,
        discrete_outputs = FLAGS.discrete_outputs,
        batch_size = FLAGS.batch_size,
        episode_len = FLAGS.episode_len,
        num_hops = FLAGS.num_hops,
        num_clusters = FLAGS.num_clusters,
        memory_size = FLAGS.memory_size,
        code_size = FLAGS.code_size,
        opt_iters = FLAGS.opt_iters,
        sr_alpha = FLAGS.sr_alpha,
        sr_beta = FLAGS.sr_beta,
        sr_gamma = FLAGS.sr_gamma,
        sr_delta = FLAGS.sr_delta,
        sr_epsilon = FLAGS.sr_epsilon,
        enc_blocks = FLAGS.enc_blocks,
        dec_blocks = FLAGS.dec_blocks,
        num_filters = FLAGS.num_filters,
        lr = FLAGS.lr,
        epochs = FLAGS.epochs)

    ## dataset
    ds_train, ds_val, ds_test = data.get_dataset(name=FLAGS.dataset, hps=hps)

    ## model and session
    model = VBM.VariationalBayesianMemory(hps)
    sess = tf.Session()

    ## tensorboard
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    ## checkpointing
    saver = tf.train.Saver()

    ## init op
    init_op = tf.global_variables_initializer()
    _ = sess.run(init_op)

    ## restoring
    if FLAGS.load_checkpoint != '' and os.path.exists(FLAGS.load_checkpoint):
        saver.restore(sess, FLAGS.load_checkpoint)
    else:
        print('load checkpoint "{}" does not exist.'.format(FLAGS.load_checkpoint))
        print('continue anyway? [y/N]')
        yn = input('> ')
        if yn.strip().lower() != 'y':
            print('program exiting.')
            return

    ## helper functions for the various modes supported by this application
    mode_to_routine = {
        'train': routines.train,
        'eval': routines.evaluate,
        'generate': routines.generate,
        'reconstruct': routines.reconstruct,
        'iterative_read': routines.iterative_read,
        'copy': routines.copy
    }
    routine = mode_to_routine[FLAGS.mode]

    ## rather than pass around tons of arguments,
    #  just use callbacks to perform the required functionality
    if FLAGS.mode == 'train':
        checkpoint_dir = FLAGS.checkpoint_dir
        callbacks = {
            'tensorboard': calls.tensorboard(train_writer), 
            'checkpointing': calls.checkpointing(sess, saver, checkpoint_dir)
        }
        routines.train(ds_train, ds_val, sess, model, callbacks, epochs=hps.epochs)

    elif FLAGS.mode == 'eval':
        routines.evaluate(ds_test, sess, model)

    else:
        output_dir = FLAGS.output_dir
        callbacks = {
            'save_png': calls.save_png(output_dir),
            'save_gif': calls.save_gif(output_dir)
        }
        routine(ds_test, sess, model, callbacks)


if __name__ == '__main__':
    tf.app.run()
