"""
Title: 3D volumetric rendering with NeRF
Authors: [Aritra Roy Gosthipaty](https://twitter.com/arig23498), [Ritwik Raha](https://twitter.com/ritwik_raha)
Date created: 2021/08/09
Last modified: 2021/08/09
Description: Minimal implementation of volumetric rendering as shown in NeRF.
"""
"""
## Introduction

In this example, we present a minimal implementation of the research paper
[**NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**](https://arxiv.org/abs/2003.08934)
by Ben Mildenhall et. al. The authors have proposed an ingenious way
to *synthesize novel views of a scene* by modelling the *volumetric
scene function* through a neural network.

To help you understand this intuitively, let's start with the following question:
*would it be possible to give to a neural
network the position of a pixel in an image, and ask the network
to predict the color at that position?*

| ![2d-train](https://i.imgur.com/DQM92vN.png) |
| :---: |
| **Figure 1**: A neural network being given coordinates of an image
as input and asked to predict the color at the coordinates. |

The neural network would hypothetically *memorize* (overfit on) the
image. This means that our neural network would have encoded the entire image
in its weights. We could query the neural network with each position,
and it would eventually reconstruct the entire image.

| ![2d-test](https://i.imgur.com/6Qz5Hp1.png) |
| :---: |
| **Figure 2**: The trained neural network recreates the image from scratch. |

A question now arises, how do we extend this idea to learn a 3D
volumetric scene? Implementing a similar process as above would
require the knowledge of every voxel (volume pixel). Turns out, this
is quite a challenging task to do.

The authors of the paper propose a minimal and elegant way to learn a
3D scene using a few images of the scene. They discard the use of
voxels for training. The network learns to model the volumetric scene,
thus generating novel views (images) of the 3D scene that the model
was not shown at training time.

There are a few prerequisites one needs to understand to fully
appreciate the process. We structure the example in such a way that
you will have all the required knowledge before starting the
implementation.
"""

"""
## Setup
"""

# Setting random seed to obtain reproducible results.
import tensorflow as tf

tf.random.set_seed(42)

import os
import glob
import imageio
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from load_llff import load_llff_data
from utils import *

class NeRF(keras.Model):
    def __init__(self, nerf_model, batch_size, num_samples, h, w):
        super().__init__()
        self.nerf_model = nerf_model
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.h = h
        self.w = w

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_metric = keras.metrics.Mean(name="psnr")

    def train_step(self, inputs):
        # Get the images and the rays.
        (images, rays) = inputs
        (rays_flat, t_vals) = rays

        with tf.GradientTape() as tape:
            # Get the predictions from the model.
            rgb, _ = render_rgb_depth(
                model=self.nerf_model,
                rays_flat=rays_flat,
                t_vals=t_vals,
                batch_size=self.batch_size,
                num_samples=self.num_samples,
                h=self.h,
                w=self.w,
                rand=True
            )
            loss = self.loss_fn(images, rgb)

        # Get the trainable variables.
        trainable_variables = self.nerf_model.trainable_variables

        # Get the gradeints of the trainiable variables with respect to the loss.
        gradients = tape.gradient(loss, trainable_variables)

        # Apply the grads and optimize the model.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # Get the PSNR of the reconstructed images and the source images.
        psnr = tf.image.psnr(images, rgb, max_val=1.0)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}

    def test_step(self, inputs):
        # Get the images and the rays.
        (images, rays) = inputs
        (rays_flat, t_vals) = rays

        # Get the predictions from the model.
        rgb, _ = render_rgb_depth(
            model=self.nerf_model,
            rays_flat=rays_flat,
            t_vals=t_vals,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            h=self.h,
            w=self.w,
            rand=True
        )
        loss = self.loss_fn(images, rgb)

        # Get the PSNR of the reconstructed images and the source images.
        psnr = tf.image.psnr(images, rgb, max_val=1.0)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.psnr_metric]

class TrainMonitor(keras.callbacks.Callback):
    def __init__(self, epochs, test_rays_flat, test_t_vals, batch_size, num_samples, h, w) -> None:
        super(TrainMonitor, self).__init__()
        self.epochs = epochs
        self.test_rays_flat = test_rays_flat
        self.test_t_vals = test_t_vals
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.h = h
        self.w = w

    def on_train_begin(self, logs=None):
        self.loss_list = []

    def on_epoch_end(self, epoch, logs=None):
        loss = logs["loss"]
        self.loss_list.append(loss)
        test_recons_images, depth_maps = render_rgb_depth(
            model=self.model.nerf_model,
            rays_flat=self.test_rays_flat,
            t_vals=self.test_t_vals,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            w=self.w,
            h=self.h,
            rand=True,
            train=False,
        )

        # Plot the rgb, depth and the loss plot.
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        ax[0].imshow(keras.preprocessing.image.array_to_img(test_recons_images[0]))
        ax[0].set_title(f"Predicted Image: {epoch:03d}")

        ax[1].imshow(keras.preprocessing.image.array_to_img(depth_maps[0, ..., None]))
        ax[1].set_title(f"Depth Map: {epoch:03d}")

        ax[2].plot(self.loss_list)
        ax[2].set_xticks(np.arange(0, self.epochs + 1, 5.0))
        ax[2].set_title(f"Loss Plot: {epoch:03d}")

        fig.savefig(f"images/{epoch:03d}.png")
        plt.show()
        plt.close()

class Inference:
    def __init__(self, batch_size=5, num_samples=32, pos_encode_dims=16, epochs=20, file_name='tiny_nerf_data.npz', url='https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz') -> None:
        AUTO = tf.data.AUTOTUNE
        BATCH_SIZE = batch_size
        NUM_SAMPLES = num_samples
        POS_ENCODE_DIMS = pos_encode_dims
        EPOCHS = epochs

        # Download the data if it does not already exist.
        if not os.path.exists(file_name) and 'http' in url:
            data = keras.utils.get_file(fname=file_name, origin=url)
        else:
            data = file_name

        data = np.load(data)
        images = data["images"]
        im_shape = images.shape
        (num_images, H, W, _) = images.shape
        (poses, focal) = (data["poses"], data["focal"])

        # Plot a random image from the dataset for visualization.
        plt.imshow(images[np.random.randint(low=0, high=num_images)])
        plt.show()

        # Create the training split.
        split_index = int(num_images * 0.8)

        # Split the images into training and validation.
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Split the poses into training and validation.
        train_poses = poses[:split_index]
        val_poses = poses[split_index:]

        # Make the training pipeline.
        train_img_ds = tf.data.Dataset.from_tensor_slices(train_images)
        train_pose_ds = tf.data.Dataset.from_tensor_slices(train_poses)
        train_ray_ds = train_pose_ds.map(lambda pose: map_fn(pose, H, W, focal, NUM_SAMPLES, POS_ENCODE_DIMS), num_parallel_calls=AUTO)
        training_ds = tf.data.Dataset.zip((train_img_ds, train_ray_ds))
        train_ds = (
            training_ds.shuffle(BATCH_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
            .prefetch(AUTO)
        )

        # Make the validation pipeline.
        val_img_ds = tf.data.Dataset.from_tensor_slices(val_images)
        val_pose_ds = tf.data.Dataset.from_tensor_slices(val_poses)
        val_ray_ds = val_pose_ds.map(lambda pose: map_fn(pose, H, W, focal, NUM_SAMPLES, POS_ENCODE_DIMS), num_parallel_calls=AUTO)
        validation_ds = tf.data.Dataset.zip((val_img_ds, val_ray_ds))
        val_ds = (
            validation_ds.shuffle(BATCH_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
            .prefetch(AUTO)
        )


        test_imgs, test_rays = next(iter(train_ds))
        test_rays_flat, test_t_vals = test_rays

        loss_list = []


        num_pos = H * W * NUM_SAMPLES
        nerf_model = get_nerf_model(num_layers=8, num_pos=num_pos, pos_encode_dims=POS_ENCODE_DIMS)

        model = NeRF(nerf_model, BATCH_SIZE, NUM_SAMPLES, H, W)
        model.compile(
            optimizer=keras.optimizers.Adam(), loss_fn=keras.losses.MeanSquaredError()
        )

        # Create a directory to save the images during training.
        if not os.path.exists("images"):
            os.makedirs("images")

        model.fit(
            train_ds,
            validation_data=val_ds,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[TrainMonitor(EPOCHS, test_rays_flat, test_t_vals, BATCH_SIZE, NUM_SAMPLES, H, W)],
            steps_per_epoch=split_index // BATCH_SIZE,
        )


        create_gif("images/*.png", "training.gif")

        # Get the trained NeRF model and infer.
        nerf_model = model.nerf_model
        test_recons_images, depth_maps = render_rgb_depth(
            model=nerf_model,
            rays_flat=test_rays_flat,
            t_vals=test_t_vals,
            batch_size=BATCH_SIZE,
            num_samples=NUM_SAMPLES,
            h=H,
            w=W,
            rand=True,
            train=False,
        )

        # Create subplots.
        fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 20))

        for ax, ori_img, recons_img, depth_map in zip(
            axes, test_imgs, test_recons_images, depth_maps
        ):
            ax[0].imshow(keras.preprocessing.image.array_to_img(ori_img))
            ax[0].set_title("Original")

            ax[1].imshow(keras.preprocessing.image.array_to_img(recons_img))
            ax[1].set_title("Reconstructed")

            ax[2].imshow(
                keras.preprocessing.image.array_to_img(depth_map[..., None]), cmap="inferno"
            )
            ax[2].set_title("Depth Map")


        rgb_frames = []
        batch_flat = []
        batch_t = []

        # Iterate over different theta value and generate scenes.
        for index, theta in tqdm(enumerate(np.linspace(0.0, 360.0, 120, endpoint=False))):
            # Get the camera to world matrix.
            c2w = pose_spherical(theta, -30.0, 4.0)

            #
            ray_oris, ray_dirs = get_rays(H, W, focal, c2w)
            rays_flat, t_vals = render_flat_rays(
                ray_oris, ray_dirs, near=2.0, far=6.0, num_samples=NUM_SAMPLES, pos_encode_dims=POS_ENCODE_DIMS, rand=False
            )

            if index % BATCH_SIZE == 0 and index > 0:
                batched_flat = tf.stack(batch_flat, axis=0)
                batch_flat = [rays_flat]

                batched_t = tf.stack(batch_t, axis=0)
                batch_t = [t_vals]

                rgb, _ = render_rgb_depth(
                    nerf_model,
                    batched_flat,
                    batched_t,
                    BATCH_SIZE,
                    NUM_SAMPLES,
                    H,
                    W,
                    rand=False,
                    train=False
                )

                temp_rgb = [np.clip(255 * img, 0.0, 255.0).astype(np.uint8) for img in rgb]

                rgb_frames = rgb_frames + temp_rgb
            else:
                batch_flat.append(rays_flat)
                batch_t.append(t_vals)

        rgb_video = "rgb_video.mp4"
        imageio.mimwrite(rgb_video, rgb_frames, fps=30, quality=7, macro_block_size=None)

if __name__ == "__main__":
    Inference(epochs=20)
