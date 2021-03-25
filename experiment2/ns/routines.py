import tensorflow_datasets as tfds
import numpy as np
import collections
import matplotlib.pyplot as plt
import uuid

def train(ds_train, ds_val, sess, model, callbacks, epochs):
    step = None
    validation_elbo_per_frame = None
    stop = False
    last_peak_epoch = None

    for epoch in range(0, epochs):
        if stop:
            break

        for batch in tfds.as_numpy(ds_train):
            # batch is a batch of episodes.

            elbo_per_frame, step, summary = model.train(sess, batch)
            print('Epoch {}, Global Step {}, ELBO per frame: {}'.format(epoch, step, elbo_per_frame))

            _ = callbacks['tensorboard'](step, summary)

        print('Epoch {} is done!'.format(epoch))
        print('Evaluating on validation set...')
        new_validation_elbo_per_frame = evaluate(ds_val, sess, model)

        if last_peak_epoch is None:
            print('\tSaving checkpoint...')
            _ = callbacks['checkpointing'](step)
            validation_elbo_per_frame = new_validation_elbo_per_frame
            last_peak_epoch = 0
        else:
            if new_validation_elbo_per_frame < validation_elbo_per_frame:
                print('\tValidation set ELBO per frame of {}'.format(new_validation_elbo_per_frame))
                print('\tis worse than previous best ELBO per frame of {}'.format(validation_elbo_per_frame))

                if epoch - last_peak_epoch >= 10:
                    print('\tStopping early...')
                    stop = True

            else:
                print('\tValidation set ELBO per frame of {}'.format(new_validation_elbo_per_frame))
                print('\tis better than previous best ELBO per frame of {}'.format(validation_elbo_per_frame))
                print('\tSaving checkpoint...')
                _ = callbacks['checkpointing'](step)
                validation_elbo_per_frame = new_validation_elbo_per_frame
                last_peak_epoch = epoch


def evaluate(dataset, sess, model):
    elbo_per_frame_sum = 0.0
    batch_count = 0

    for batch in tfds.as_numpy(dataset):
        # batch is a batch of episodes.

        elbo_per_frame = model.evaluate(sess, batch)
        elbo_per_frame_sum += elbo_per_frame
        batch_count += 1

    elbo_per_frame_avg = elbo_per_frame_sum / float(batch_count)
    print('ELBO per frame: {}'.format(elbo_per_frame_avg))
    return elbo_per_frame_avg


def generate(dataset, sess, model, callbacks):
    # This method generates new frames conditioned on a context state.
    # This context state is a distribution, which will be sampled from: int q(c)p(z|c)p(x|z) dzdc.

    for batch in tfds.as_numpy(dataset):
        # batch is a batch of episodes.

        episode = batch[0]
        batch = np.expand_dims(episode, 0)

        context_state = model.get_context_state(sess, batch)

        # visualization of generation process will show the frames from an episode
        # arrange these into one row
        viz_episode = np.concatenate(episode, axis=1)

        # now get some generated frames conditioned on context state
        # the visualization will display a grid of these generated frames
        rows = []
        for i in range(0, 16):
            row = []
            for j in range(0, 16):
                img = model.generate_from_context_state(sess, context_state)
                img = np.squeeze(img, 0)
                row.append(img)
            row = np.concatenate(row, axis=1)
            rows.append(row)
        rows = np.concatenate(rows, axis=0)  # make a square

        # pad the visualization with whitespace.
        whitespace_episode = np.ones(dtype=np.float32, shape=viz_episode.shape)
        whitespace_rest = np.ones(dtype=np.float32,
                                  shape=(rows.shape[0], viz_episode.shape[1] - rows.shape[1], rows.shape[-1]))

        viz1 = np.concatenate([viz_episode, whitespace_episode], axis=0)
        viz2 = np.concatenate([rows, whitespace_rest], axis=1)
        img = np.concatenate([viz1, viz2], axis=0)

        if img.shape[-1] == 3:
            img = 1.0 * img #0.5 * img + 0.5
        else:
            img = np.concatenate([img for _ in range(0, 3)], axis=-1)

        print('saving images...')
        fp = callbacks['save_png'](img)
        print(fp)
        break


def reconstruct(dataset, sess, model, callbacks):
    for batch in tfds.as_numpy(dataset):

        episode = batch[0]
        batch = np.expand_dims(episode, 0)

        context_state = model.get_context_state(sess, batch)

        episode_len = episode.shape[0]

        read_xs = []
        for i in range(0, episode_len):
            x = batch[:, i]  # [1, h, w, c]
            img = model.read_from_context_state(sess, context_state, x)
            img = np.squeeze(img, 0)
            read_xs.append(img)

        original = np.concatenate(episode, axis=1)
        reconstructed = np.concatenate(read_xs, axis=1)
        img = np.concatenate([original, reconstructed], axis=0)

        if img.shape[-1] == 3:
            img = 1.0 * img #0.5 * img + 0.5
        else:
            img = np.concatenate([img for _ in range(0, 3)], axis=-1)

        print('saving images...')
        fp = callbacks['save_png'](img)
        print(fp)
        break

