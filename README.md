# Port of neural-style to PyTorch

Port of [my fork](https://github.com/avoroshilov/neural-style) of the TensorFlow implementation of the "A Neural Algorithm of Artistic Style" paper.

WIP. Currently implemented:
* Skeleton of the style transfer.
* Total Variation denoising loss.
* Style distribution loss.

TODO:
* Hierarchical style transfer.
* Various small tweaks like style exponential weighting, and similar.
* Activation shift.
* Color preservation.
* Collaging.
* CLI (currently everything is hardcoded).

To run, by default you need to have a VGG19 checkpoint at `vgg/vgg19-imagenet.pth`.
If you don't have one, you can either download https://download.pytorch.org/models/vgg19-dcbb9e9d.pth manually and rename, or simply set `weights_path=None` in `vgg.get_VGG_features` in [stylize.py][stylize.py]. In this case, PyTorch will download the checkpoint automatically and cache it.
