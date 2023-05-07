import os
import time
import random

import PIL.Image
import numpy as np
import torch
import torchvision
import torchvision.models
import torchvision.transforms
import torch.utils.data.sampler


import stylize
import vgg

RANDOM_SEED = 1337
def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg.calc_VGG_normalization_tensors(device)

    VGG19_features = vgg.get_VGG_features(device, weights_path="vgg/vgg19-imagenet.pth")
    VGG19_processed = vgg.rebuild_VGG(VGG19_features)

    def load_resize_image(image_path, target_size=None):
        # Open image dropping alpha where needed
        image = PIL.Image.open(image_path).convert("RGB")
        image_size = image.size
        print(f"{image.size=}")

        if target_size is None:
            # Make image square
            min_side = min(image_size[0], image_size[1])
            target_size = (min_side, min_side)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(target_size),
            torchvision.transforms.ToTensor(),
        ])
        image = transforms(image)

        # We need this unsqueeze to add batch dimension to match what VGG expects
        image = image.unsqueeze(0)

        return image.to(device, torch.float), image_size

    CONTENT_IMG_PATH = "cat_sm.png"
    content_img, original_size = load_resize_image(CONTENT_IMG_PATH)
    print(f"{original_size=}")
    print(f"{content_img.shape=}")

    rescaled_size = (content_img.shape[2], content_img.shape[3])
    print(f"{rescaled_size=}")
    STYLE_IMG_PATH = "s10_starryNight_big.jpg"
    style_img, style_size = load_resize_image(STYLE_IMG_PATH, rescaled_size)

    # Convert the content image into the VGG normalized space, and run VGG over it to generate features
    processed_content = vgg.preprocess(content_img)
    content_features = stylize.extract_features(processed_content, VGG19_processed, vgg.CONTENT_LAYERS)
    for feat_name in content_features:
        content_features[feat_name] = content_features[feat_name].detach()

    # Convert the style image into the VGG normalized space, and run VGG over it to generate features
    processed_style = vgg.preprocess(style_img)
    style_features = stylize.extract_features(processed_style, VGG19_processed, vgg.STYLE_LAYERS)
    for feat_name in style_features:
        style_features[feat_name] = style_features[feat_name].detach()

    # Process style features into derivative features used for style transfer
    style_features_gram, style_features_distr = stylize.get_processed_style_features(style_features)

    set_random_seeds(RANDOM_SEED)

    # Create initial image to start style transfer from
    NOISEBLEND_COEFF = 0.65
    noise_tensor = torch.randn(processed_content.shape).to(device)
    processed_content_optim = (1.0 - NOISEBLEND_COEFF) * processed_content.clone() + NOISEBLEND_COEFF * noise_tensor
    processed_content_optim.requires_grad_(True)
    VGG19_processed.requires_grad_(False)


    def get_filename_without_extension(path):
        filename, ext = os.path.splitext(path)
        return filename

    USE_LBFGS = True
    # Default = 100 # This is PyTorch's default, and it was the best vs wallclock time in my experiments
    # Short = 200
    # Medium = 500
    # Long = 700
    # XL = inf (let it store history for each step)
    #LBFGS_MAX_HISTORRY_SIZE = 100 # This is PyTorch's default, and it was the best vs wallclock time in my experiments
    LBFGS_MAX_HISTORRY_SIZE = 2 # This is PyTorch's default, and it was the best vs wallclock time in my experiments

    optimizer_name = ""
    if USE_LBFGS:
        optimizer_name = f"L-BFGS_{LBFGS_MAX_HISTORRY_SIZE}"
    else:
        optimizer_name = "AdamW"

    content_img_filename = get_filename_without_extension(CONTENT_IMG_PATH)
    style_img_filename = get_filename_without_extension(STYLE_IMG_PATH)
    tb_run_name = f"{content_img_filename}__{style_img_filename}__{RANDOM_SEED}__{optimizer_name}__{int(time.time())}"


    stylize.stylize(
        VGG19_processed,
        processed_content_optim,
        content_features,
        style_features_gram,
        style_features_distr,
        USE_LBFGS,
        LBFGS_MAX_HISTORRY_SIZE,
        tb_run_name,
    )

    result = vgg.unprocess(processed_content_optim.clone())
    result.clamp_(0.0, 1.0)

    def save_result(result_tensor, filename="test_result.png"):
        untransforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            # For whatever reason torchvision messes up parameter ordering
            torchvision.transforms.Resize((original_size[1], original_size[0])),
        ])


        result_PIL = untransforms(result_tensor.detach().clone().squeeze(0))
        result_PIL.save(filename)

    save_result(result, "test_result.png")

if __name__ == '__main__':
    main()
