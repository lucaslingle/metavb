import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds
_ = tf.logging.set_verbosity(tf.logging.ERROR)

from dynamic_kanerva_machine import DynamicKanervaMachine
import data
import callbacks as calls
import routines as routines
import os

flags = tf.app.flags


flags.DEFINE_enum(
    "mode", 'train', ['train', 'eval', 'generate', 'reconstruct', 'iterative_read', 'copy'], 
    "mode: one of train, eval, generate, reconstruct, iterative_read, copy")

flags.DEFINE_enum("dataset", 'mnist', ['mnist', 'celeb_a', 'cifar10', 'omniglot'], "dataset: which dataset to use")
flags.DEFINE_integer("img_height", 28, "img_height: height to scale images to, in pixels")
flags.DEFINE_integer("img_width", 28, "img_width: width to scale images to, in pixels")
flags.DEFINE_integer("img_channels", 1, "img_channels: number of image channels")
flags.DEFINE_boolean("discrete_outputs", True, "discrete_outputs: affects data preproc and gen dist over pixels")

flags.DEFINE_integer("batch_size", 16, "batch_size: number of episodes per minibatch")
flags.DEFINE_integer("episode_len", 64, "episode_len: number of observations per episode")
flags.DEFINE_integer("memory_size", 32, "memory_size: number of memory rows")
flags.DEFINE_integer("code_size", 100, "code_size: dimension of a code vector/memory row")

flags.DEFINE_integer("enc_blocks", 3, "enc_blocks: number of blocks in the encoder")
flags.DEFINE_integer("dec_blocks", 3, "dec_blocks: number of blocks in the decoder")
flags.DEFINE_integer("num_filters", 16, "num_filters: number of convolutional filters per layer")

flags.DEFINE_boolean("sample_memory", True, "sample_memory: sample memory during training and evaluation")
flags.DEFINE_boolean("trainable_memory", False, "trainable_memory: use a trainable memory prior")
flags.DEFINE_boolean("use_bn", True, "use_bn: use batch normalization in residual blocks")

flags.DEFINE_float("lr", 0.001, "lr: learning rate for Adam optimizer")

flags.DEFINE_string("summaries_dir", 'tensorboard_logs/km_summaries/', "summaries_dir: directory for tensorboard logs")
flags.DEFINE_string("output_dir", 'output/', "output_dir: directory for visualizations")

flags.DEFINE_string("checkpoint_dir", 'checkpoints/', "checkpoint_dir: directory for saving model checkpoints")
flags.DEFINE_string("load_checkpoint", '', "load_checkpoint: checkpoint directory or checkpoint to load")
flags.DEFINE_integer("checkpoint_frequency", 95, "checkpoint_frequency: frequency to save checkpoints, measured in global steps")

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
        memory_size = FLAGS.memory_size,
        code_size = FLAGS.code_size,
        enc_blocks = FLAGS.enc_blocks,
        dec_blocks = FLAGS.dec_blocks,
        num_filters = FLAGS.num_filters,
        sample_memory = FLAGS.sample_memory,
        trainable_memory = FLAGS.trainable_memory,
        use_bn = FLAGS.use_bn,
        lr = FLAGS.lr,
        epochs = FLAGS.epochs)

    ## dataset
    ds_train, ds_val, ds_test = data.get_dataset(name=FLAGS.dataset, hps=hps)

    ## model and session
    model = DynamicKanervaMachine(hps)
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
        checkpoint_frequency = FLAGS.checkpoint_frequency
        callbacks = {
            'tensorboard': calls.tensorboard(train_writer), 
            'checkpointing': calls.checkpointing(sess, saver, checkpoint_dir)
        }
        routines.train(ds_train, ds_val, sess, model, callbacks, epochs=hps.epochs)

    elif FLAGS.mode == 'eval':
        callbacks = {}
        routine(ds_test, sess, model)

    else:
        output_dir = FLAGS.output_dir
        callbacks = {
            'save_png': calls.save_png(output_dir),
            'save_gif': calls.save_gif(output_dir)
        }
        routine(ds_test, sess, model, callbacks)


if __name__ == '__main__':
    tf.app.run()
