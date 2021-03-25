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
    # This method generates new frames conditioned on a memory state.
    # This memory state is a distribution. 
    # It can either be the variational posterior q(M) from a real episode X, or it can be the prior p(M). 
    # We obtain M itself by sampling the distribution or by taking the mean.

    for batch in tfds.as_numpy(dataset):
        # batch is a batch of episodes.

        episode = batch[0]
        batch = np.expand_dims(episode, 0)

        #memory_state, s1_probs, s2_probs_given_s1 = model.get_posterior_memory_state(sess, batch)
        memory_state, sg_sharps = model.get_posterior_memory_state(sess, batch)

        # save a visualization of the memory state.
        #'''
        Rs_stacked = np.concatenate([np.concatenate([memory_state.R[0, g, h, :, :] for g in range(0, model.num_hops)], axis=-1) for h in range(0, model.num_clusters)], axis=0)
        Rs_stacked = np.expand_dims(Rs_stacked, -1)
        mem_img = np.concatenate([Rs_stacked, Rs_stacked, Rs_stacked], axis=-1)
        fp = callbacks['save_png'](mem_img)
        print(fp)
        #'''

        # visualization of generation process will show the frames from an episode
        # arrange these into a few rows, assuming episode length is divisible by square_len used for our grid of samples.

        square_len = 16
        assert model.episode_len % square_len == 0
        viz_episode = np.concatenate([np.concatenate(episode[i:(i+square_len)], axis=1) for i in range(0, model.episode_len, square_len)], axis=0)

        # now get some generated frames conditioned on memory state
        # the visualization will display a grid of these generated frames
        rows = []
        for i in range(0, square_len):
            row = []
            for j in range(0, square_len):
                img = model.generate_from_memory_state(sess, memory_state, sg_sharps) #s1_probs, s2_probs_given_s1)
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

        #memory_state, _, _ = model.get_posterior_memory_state(sess, batch)
        memory_state, _ = model.get_posterior_memory_state(sess, batch)
        episode_len = episode.shape[0]

        read_xs = []
        for i in range(0, episode_len):
            x = batch[:, i]  # [1, h, w, c]
            img, z = model.read_from_memory_state(sess, memory_state, x)
            norm = np.sqrt(np.sum(np.square(z)))
            print(norm)
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


def iterative_read(dataset, sess, model, callbacks, show_nearest_neighbor=True):
    for batch in tfds.as_numpy(dataset):

        episode = batch[0]
        episode_len = episode.shape[0]
        batch = np.expand_dims(episode, 0)

        #memory_state, s1_probs, s2_probs_given_s1 = model.get_posterior_memory_state(sess, batch)
        memory_state, sg_sharps = model.get_posterior_memory_state(sess, batch)

        num_progressions = 100
        progression_len = max(10, min(episode_len-1, 23)) * 2

        nn_idxs = []

        progressions = []
        viz_dilation_rate = 1
        for _ in range(0, num_progressions):
            query_xs = []
            viz_xs = []

            x = model.generate_from_memory_state(sess, memory_state, sg_sharps) #s1_probs, s2_probs_given_s1)
            query_xs.append(x)
            viz_xs.append(x)

            for iter_ in range(0, progression_len * viz_dilation_rate):
                x = query_xs[-1]
                x, _ = model.read_from_memory_state(sess, memory_state, x)
                query_xs.append(x)
                if iter_ % viz_dilation_rate == 0:
                    viz_xs.append(x)

            viz_xs = [np.squeeze(x, 0) for x in viz_xs]  # get rid of batch dim, get list of hwc shaped tensors

            last = viz_xs[-1]
            nearest_neighbor_idx = np.argmin([np.sum(np.square(last - x_i)) for x_i in episode]) # L2 norm
            #nearest_neighbor_idx = np.argmin([np.sum(np.abs(last - x_i)) for x_i in episode]) # L1 norm
            nn_idxs.append(nearest_neighbor_idx)

            if show_nearest_neighbor:
                nearest_neighbor = episode[nearest_neighbor_idx]
                blank_space = np.ones_like(nearest_neighbor)
                viz_xs.append(blank_space)
                viz_xs.append(nearest_neighbor)

            progression = np.concatenate(viz_xs, axis=1)
            progressions.append(progression)

        progressions = np.concatenate(progressions, axis=0)

        episode_row = np.concatenate(episode, axis=1)

        print('uniques retrieved: {}'.format(len(set(nn_idxs))))

        # if the progression lengths don't match the episode length, pad the shorter of the two.
        if episode_row.shape[1] - progressions.shape[1] > 0:
            height = progressions.shape[0]
            width = episode_row.shape[1] - progressions.shape[1]
            channels = progressions.shape[-1]
            whitespace = np.ones(
                dtype=np.float32, shape=(height, width, channels))
            progressions = np.concatenate([progressions, whitespace], axis=1)

        elif progressions.shape[1] - episode_row.shape[1] > 0:
            height = episode_row.shape[0]
            width = progressions.shape[1] - episode_row.shape[1]
            channels = progressions.shape[-1]
            whitespace = np.ones(
                dtype=np.float32, shape=(height, width, channels))
            episode_row = np.concatenate([episode_row, whitespace], axis=1)

        whitespace_row = np.ones_like(episode_row)
        img = np.concatenate([episode_row, whitespace_row, progressions], axis=0)

        if img.shape[-1] == 3:
            img = 1.0 * img #0.5 * img + 0.5
        else:
            img = np.concatenate([img for _ in range(0, 3)], axis=-1)

        print('saving images...')
        fp = callbacks['save_png'](img)
        print('iterative read progressions:')
        print(fp)

        break


def copy(dataset, sess, model, callbacks):
    for batch in tfds.as_numpy(dataset):

        episode = batch[0]
        batch = np.expand_dims(episode, 0)

        episode_len = episode.shape[0]

        copied_xs = []
        for i in range(0, episode_len):
            x = batch[:, i]  # [1, h, w, c]
            img, z = model.copy(sess, x)
            norm = np.sqrt(np.sum(np.square(z)))
            print(norm)
            img = np.squeeze(img, 0)
            copied_xs.append(img)

        original = np.concatenate(episode, axis=1)
        copied = np.concatenate(copied_xs, axis=1)
        img = np.concatenate([original, copied], axis=0)

        if img.shape[-1] == 3:
            img = 1.0 * img #0.5 * img + 0.5
        else:
            img = np.concatenate([img for _ in range(0, 3)], axis=-1)

        print('saving images...')
        fp = callbacks['save_png'](img)
        print(fp)
        break


