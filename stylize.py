import time

import torch
import torch.utils.tensorboard

import vgg

def extract_features(input, named_model, feature_names, DEBUG_PRINT=False):
    feature_names_list = list(feature_names)
    features = {}
    for layer_name, layer in named_model.named_children():
        if DEBUG_PRINT:
            print(f"'{layer_name}': {layer}")

        input = layer(input)
        if layer_name in feature_names_list:
            features[layer_name] = input
            feature_names_list.remove(layer_name)


        if not feature_names_list:
            # Don't need to evaluate layers after the last one
            if DEBUG_PRINT:
                print(f"Last feature: {layer_name}")
            break
    return features

def build_feature_gram_matrix(input):
    BATCH_COUNT, FEAT_COUNT, FEAT_W, FEAT_H = input.size()
    features = input.view(BATCH_COUNT * FEAT_COUNT, FEAT_W * FEAT_H)

    gram_matrix = torch.mm(features, features.t())
    return gram_matrix

# Detach style features, and calculate Gram matrices for them, as well as distribution characteristics
def get_processed_style_features(style_features):
    style_features_gram = {}
    style_features_distr = {}
    for feat_name in style_features:
        cur_style_feature = style_features[feat_name]

        # Prepare Gram matrices for Style Loss
        style_features_gram[feat_name] = build_feature_gram_matrix(cur_style_feature)

        # Prepare distributions for Distribution Loss
        BATCH_COUNT, FEAT_COUNT, FEAT_W, FEAT_H = cur_style_feature.size()
        cur_style_feature_flatten = cur_style_feature.view(BATCH_COUNT, FEAT_COUNT, -1)
        style_features_distr[feat_name] = {
            "mean": torch.mean(cur_style_feature_flatten, dim=2, keepdim=True),
            "std": torch.std(cur_style_feature_flatten, dim=2, keepdim=True),
        }

    return style_features_gram, style_features_distr

def stylize(
    VGG19_processed,
    processed_content_optim,    # initial image
    content_features,
    style_features_gram,
    style_features_distr,
    USE_LBFGS = True,
    LBFGS_MAX_HISTORRY_SIZE = 100,
    tb_writer_run_name = None,
    tb_writer = None,   # existing TensorBoard writer
):
    NUM_OPTIM_STEPS = 4000

    optimizer_name = ""
    if USE_LBFGS:
        num_steps = 1
        optimizer = torch.optim.LBFGS(
            [processed_content_optim],
            max_eval=15000,
            max_iter=NUM_OPTIM_STEPS,
            history_size=min(LBFGS_MAX_HISTORRY_SIZE,NUM_OPTIM_STEPS),
            #tolerance_grad=1e-5,
            #tolerance_change=1e-9,
        )
        optimizer_name = f"L-BFGS_{LBFGS_MAX_HISTORRY_SIZE}"
    else:
        # Try and match L-BFGS execution time
        num_steps = int(1.35*NUM_OPTIM_STEPS)
        optimizer = torch.optim.AdamW([processed_content_optim], lr=1e-1)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_steps)
        optimizer_name = "AdamW"

    if tb_writer is None and tb_writer_run_name is not None:
        # content_img_filename = get_filename_without_extension(CONTENT_IMG_PATH)
        # style_img_filename = get_filename_without_extension(STYLE_IMG_PATH)
        # run_name = f"{content_img_filename}__{style_img_filename}__{RANDOM_SEED}__{optimizer_name}__{int(time.time())}"
        # tb_writer = torch.utils.tensorboard.SummaryWriter(f"tb_runs/{run_name}")
        tb_writer = torch.utils.tensorboard.SummaryWriter(f"tb_runs/{tb_writer_run_name}")

    BATCH_COUNT, CHANNEL_COUNT, IMG_W, IMG_H = processed_content_optim.shape

    CONTENT_LAYERS = vgg.get_content_layers()
    STYLE_LAYERS = vgg.get_style_layers()
    COMBINED_LAYERS = [*set(CONTENT_LAYERS + STYLE_LAYERS)]

    total_step = 0

    STYLE_WEIGHT = 1e6
    TV_WEIGHT = 3.0
    STYLE_DISTR_WEIGHT = 30.0

    start_time = time.time()
    for step in range(num_steps):
        def lbgfs_closure():
            optimizer.zero_grad()
            combined_features = extract_features(processed_content_optim, VGG19_processed, COMBINED_LAYERS)

            content_loss = 0
            for layer_name in CONTENT_LAYERS:
                content_loss += torch.nn.functional.mse_loss(combined_features[layer_name], content_features[layer_name])

            style_distribution_loss = 0
            style_loss = 0
            for layer_name in STYLE_LAYERS:
                cur_style_feature = combined_features[layer_name]
                BATCH_COUNT, FEAT_COUNT, FEAT_W, FEAT_H = cur_style_feature.shape

                if STYLE_WEIGHT != 0.0:
                    # Regular Style Loss from Gatys et al.
                    # From the paper directly, contribution of Gram matrices is multiplied by
                    #   1 / (4 * Nl^2 * Ml^2)
                    # where Nl - number of layer feature maps, and Ml is height times width of a feature map
                    Nl = FEAT_COUNT
                    Ml = FEAT_W * FEAT_H
                    gram = build_feature_gram_matrix(cur_style_feature)
                    style_loss += 1.0 / (4.0 * Nl*Nl * Ml*Ml) * torch.nn.functional.mse_loss(gram, style_features_gram[layer_name])

                if STYLE_DISTR_WEIGHT != 0.0:
                    # Distribution Loss
                    cur_style_feature_flatten = cur_style_feature.view(BATCH_COUNT, FEAT_COUNT, -1)

                    cur_style_feat_distr = {
                        "mean": torch.mean(cur_style_feature_flatten, dim=2, keepdim=True),
                        "std": torch.std(cur_style_feature_flatten, dim=2, keepdim=True),
                    }
                    target_style_feat_distr = style_features_distr[layer_name]

                    USE_PERFEATURE_DISTR_LOSS = False
                    if USE_PERFEATURE_DISTR_LOSS:
                        # Remapping features to a target distribution
                        cur_style_normalized_feature_flatten = (cur_style_feature_flatten - cur_style_feat_distr["mean"]) / (cur_style_feat_distr["std"] + 1e-6)
                        rematched_features_flatten = target_style_feat_distr["std"] * cur_style_normalized_feature_flatten + target_style_feat_distr["mean"]

                        style_distribution_loss += torch.nn.functional.mse_loss(rematched_features_flatten, cur_style_feature_flatten)
                    else:
                        # Just using the distribution parameters directly
                        style_distribution_loss += torch.nn.functional.mse_loss(cur_style_feat_distr["mean"], target_style_feat_distr["mean"])
                        #style_distribution_loss += torch.nn.functional.mse_loss(cur_style_feat_distr["std"], target_style_feat_distr["std"])

            style_loss *= STYLE_WEIGHT
            style_distribution_loss *= STYLE_DISTR_WEIGHT

            # Not using simplified anisotropic Total Variation loss (no powers and square roots).
            # Using mean instead of sum.
            total_variation_loss = 0
            if TV_WEIGHT != 0.0:
                tv_x = processed_content_optim[:,:,1:,:] - processed_content_optim[:,:,:-1,:]
                tv_y = processed_content_optim[:,:,:,1:] - processed_content_optim[:,:,:,:-1]
                total_variation_loss = (torch.sum(torch.pow(tv_x, 2)) + torch.sum(torch.pow(tv_y, 2))) / (BATCH_COUNT * CHANNEL_COUNT * IMG_W * IMG_H)
            total_variation_loss *= TV_WEIGHT

            total_loss = content_loss + style_loss + total_variation_loss + style_distribution_loss
            total_loss.backward()

            nonlocal start_time
            nonlocal total_step
            if total_step % 50 == 0:
                style_loss_print = 0
                if not isinstance(style_loss, int):
                    style_loss_print = style_loss.item()
                style_distribution_loss_print = 0
                if not isinstance(style_distribution_loss, int):
                    style_distribution_loss_print = style_distribution_loss.item()
                total_variation_loss_print = 0
                if not isinstance(total_variation_loss, int):
                    total_variation_loss_print = total_variation_loss.item()
                print(f"{time.time()-start_time:.2f} Sub-step {total_step}: C={content_loss.item()}, S={style_loss_print}, SD={style_distribution_loss_print}, TV={total_variation_loss_print}")
            total_step += 1

            training_time = time.time() - start_time
            if tb_writer:
                training_time_ms = training_time*1000.0
                tb_writer.add_scalar("losses/content", content_loss.item(), training_time_ms)
                tb_writer.add_scalar("losses/style", style_loss.item(), training_time_ms)
                tb_writer.add_scalar("losses/style_distr", style_distribution_loss.item(), training_time_ms)
                tb_writer.add_scalar("losses/tv_denoise", total_variation_loss.item(), training_time_ms)
                tb_writer.add_scalar("losses/total", total_loss.item(), training_time_ms)


            return total_loss
        optimizer.step(lbgfs_closure)
        if not USE_LBFGS:
            scheduler.step()
