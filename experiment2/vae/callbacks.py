import uuid
import os
import matplotlib.pyplot as plt
import numpy as np
import moviepy.editor as mpy


def checkpointing(sess, saver, checkpoint_dir):
    def func(step):
        # with early stopping I've disabled the "checkpoint frequency" setting.
        saver.save(sess, checkpoint_dir)

    return func


def tensorboard(writer):
    def func(step, summary):
        writer.add_summary(summary, step)
        writer.flush()

    return func


def save_png(output_dir):
    def func(img):
        filename = str(uuid.uuid4()) + '.png'
        fp = os.path.join(output_dir, filename)
        if img.shape[-1] == 1:
            img = np.concatenate([img for _ in range(0,3)], axis=-1)
        
        img = np.minimum(np.maximum(0.0, img), 1.0)
        plt.imsave(fp, img)
        return fp

    return func


def save_gif(output_dir):
    duration = 2
    def func(imgs):
        def make_frame(t):
            x = imgs[int(len(imgs)/duration*t)]
            x = np.minimum(np.maximum(0.0, x), 1.0)
            return (255 * x).astype(np.int32).astype(np.uint8)

        filename = str(uuid.uuid4()) + '.gif'
        fp = os.path.join(output_dir, filename)

        clip = mpy.VideoClip(make_frame, duration=duration)
        clip.write_gif(fp, fps = len(imgs) / duration)
        return fp

    return func
