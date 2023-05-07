import torch
import torch.nn
import torchvision

VGG_normalization = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

def calc_VGG_normalization_tensors(device):
    # Converting the VGG normalization to tensors, and making it [3, 1, 1] instead of just [3]
    # to match how PyTorch stores images.
    VGG_normalization["mean_tensor"] = torch.tensor(VGG_normalization["mean"]).to(device)
    VGG_normalization["mean_tensor"] = VGG_normalization["mean_tensor"].view(-1, 1, 1)
    VGG_normalization["std_tensor"] = torch.tensor(VGG_normalization["std"]).to(device)
    VGG_normalization["std_tensor"] = VGG_normalization["std_tensor"].view(-1, 1, 1)

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

def get_VGG_features(device, weights_path=None):
    # If WEIGHTS_PATH is None, will download to `C:\Users\<uname>\.cache\torch\hub\checkpoints\vgg19-dcbb9e9d.pth`
    download_weights = (weights_path is None)
    if download_weights:
        # Let `torchvision.models` download weights.
        # Currently, it downloads VGG19 (non-BN) from this URL:
        #   https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
        weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
    else:
        # Provide the local model
        weights = None

    VGG19_model = torchvision.models.vgg19(weights=weights, init_weights=False)
    if not download_weights:
        print(f"Loading model from '{weights_path}'")
        VGG19_model.load_state_dict(torch.load(weights_path))

    # Cutoff classifier and adaptive pooling which we don't need for Style Transfer
    VGG19_features = VGG19_model.features.to(device)
    VGG19_features.eval()
    return VGG19_features

def rebuild_VGG(VGG19_features):
    # Now matching the structure we had in the TF version of StyleTransfer
    VGG19_processed = torch.nn.Sequential()
    group_id = 1
    # Theoretically, we don't need both of them, as Conv always is followed by Non-linearity.
    # But I want it that way.
    conv_id = 1
    relu_id = 1
    for layer in VGG19_features.children():
        layer_name = None
        if isinstance(layer, torch.nn.MaxPool2d):
            # Max Pooling is the group boundary
            layer_name = f"pool{group_id}"
            group_id += 1
            conv_id = 1
            relu_id = 1
        elif isinstance(layer, torch.nn.Conv2d):
            layer_name = f"conv{group_id}_{conv_id}"
            conv_id += 1
        elif isinstance(layer, torch.nn.ReLU):
            layer_name = f"relu{group_id}_{relu_id}"
            # Replace the ReLU layer with out-of-place variation.
            # By default, torchvision's VGG model comes with in-place ReLU, which
            # is not good for Style Transfer experiments, e.g. if we want to use
            # Conv2d features directly.
            layer = torch.nn.ReLU(inplace=False)
            relu_id += 1
        elif isinstance(layer, torch.nn.BatchNorm2d):
            # Just in case we'll hit the BN version of VGG.
            # Don't want to count BNs separately, so just reuse the ReLU counter,
            # BNs go before the ReLUs (at least in the torchvisiion version I am using).
            layer_name = f'bn_{relu_id}'
        else:
            print(f"Unexpected layer: {layer.__class__.__name__}")
            layer_name = "<ERROR>"
            layer = None

        if layer:
            VGG19_processed.add_module(layer_name, layer)
    return VGG19_processed

def preprocess(raw_img):
    # This is practically `torchvision.transforms.Normalize`
    print(f"{raw_img.shape=}")
    return (raw_img - VGG_normalization["mean_tensor"]) / VGG_normalization["std_tensor"]

def unprocess(normalized_img):
    return normalized_img * VGG_normalization["std_tensor"] + VGG_normalization["mean_tensor"]

def get_content_layers():
    return CONTENT_LAYERS

def get_style_layers():
    return STYLE_LAYERS
