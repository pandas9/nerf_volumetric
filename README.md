# Neural Radiance Fields Volumetric Rendering
Based on [github.com/bmild/nerf](https://www.github.com/bmild/nerf)

###### The authors of the paper propose a minimal and elegant way to learn a 3D scene using a few images of the scene. They discard the use of voxels for training. The network learns to model the volumetric scene, thus generating novel views (images) of the 3D scene that the model was not shown at training time.
[Source: Keras](https://keras.io/examples/vision/nerf/)

![nerf-volumetric](example/download.png?raw=true)

# Usage
##### Run NeRF
##### Inside nerf.py adjust with desired settings
```
if __name__ == "__main__":
    Inference(
        epochs=200,
        _file='./fern',
        data_type='llff'
    )

```

##### Data
###### Generating Data
NeRF requires poses for images, to generate poses run ```python imgs2poses.py <your_folder>``` script uses COLMAP to run structure from motion to get 6-DoF camera poses and near/far depth bounds for the scene. For installing COLMAP check [colmap.github.io/install.html](https://colmap.github.io/install.html) <br />
Inside folder location make sure to have /images folder containing all of your images that NeRF will run through. <br />

<br />

After COLMAP is finished it will output ```poses_bounds.npy``` and ```/sparse``` folder containing necessary data for NeRF. <br />

If you do not wish to use LLFF you can pass ```data_type='npz', _file='my_data.npz'``` and use .npz file containing images, poses and focal

###### Pregenerated Data

Check [Pregenerated Data](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) <br />

# Acknowledgements
https://keras.io/examples/vision/nerf/ <br />
https://arxiv.org/abs/2003.08934 <br />
https://github.com/bmild/nerf <br />
https://github.com/3b1b/manim <br />
https://www.mathworks.com/help/vision/ug/camera-calibration.html <br />
https://github.com/colmap/colmap <br />
https://github.com/fyusion/llff <br />
