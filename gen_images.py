import os
import pickle
import sys

import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

sys.path.append("/stylegan3")
import argparse

import dnnlib
import legacy
import yaml

device = torch.device("cuda")


def get_rmse_feature(feature_path):
    return np.load(feature_path)


def get_randn(shape, seed=None):
    if seed:
        rnd = np.random.RandomState(seed)
        return rnd.randn(*shape)
    else:
        return np.random.randn(*shape)


def get_length(vec):
    return np.linalg.norm(vec)


def normalize_vec(vec, unit_length, factor_length):
    orig_length = get_length(vec)
    return vec * (unit_length / orig_length) * factor_length


def get_noise_vars(G):
    print(G.synthesis.w_dim)
    noise_vars = [var for name, var in G.synthesis.w_dim if name.startswith("noise")]
    noise_pairs = list(zip(noise_vars, tflib.run(noise_vars)))
    return noise_vars


def move_vec(vec, unit_length, factor_length):
    direction_vec = get_randn(vec.shape)
    direction_vec = normalize_vec(direction_vec, unit_length, factor_length)
    return vec + direction_vec


def init_latents(G):
    latents = get_randn((1, G.input_shape[1]))
    noise_vars = get_noise_vars(G)
    noise_vectors = []
    for i in range(len(noise_vars)):
        noise_vectors.append(get_randn(noise_vars[i].shape))
    return latents, noise_vectors


def gen_image(Gs, latents, noise_vectors, noise_vars, fmt, save_path="example.png"):
    for i in range(len(noise_vars)):
        tflib.set_vars({noise_vars[i]: noise_vectors[i]})
    images = Gs.run(
        latents, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt
    )

    PIL.Image.fromarray(images[0], "RGB").save(save_path)


def gen_image_mix(
    G,
    latents_high,
    latents_band,
    latents_low,
    lpf,
    hpf,
    bpf,
    save_path="example.png",
):

    high_latents = np.stack(latents_high.cpu())
    band_latents = np.stack(latents_band.cpu())
    low_latents = np.stack(latents_low.cpu())
    # high_latents = torch.stack(latents_high)
    # band_latents = torch.stack(latents_band)
    # low_latents = torch.stack(latents_low)
    high_latents = torch.from_numpy(high_latents.astype(np.float32)).clone().to(device)
    band_latents = torch.from_numpy(band_latents.astype(np.float32)).clone().to(device)
    low_latents = torch.from_numpy(low_latents.astype(np.float32)).clone().to(device)
    high_dlatents = G.mapping(high_latents, None)
    band_dlatents = G.mapping(band_latents, None)
    low_dlatents = G.mapping(low_latents, None)

    combined_dlatents = low_dlatents

    for style in lpf:
        combined_dlatents[:, style] = low_dlatents[:, style]

    for style in hpf:
        combined_dlatents[:, style] = high_dlatents[:, style]

    for style in bpf:
        combined_dlatents[:, style] = band_dlatents[:, style]

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    noise_mode = "const"  #'random'

    # print(combined_dlatents.shape)
    # combined_dlatents = torch.from_numpy(np.random.RandomState(1).randn(1, G.z_dim)).to(device)
    # seeds = [1, 2]
    # combined_dlatents = torch.from_numpy(
    #     np.stack(
    #         np.random.RandomState(1).randn(1, G.z_dim)
    #         for seed in seeds)).to(device)
    # print(combined_dlatents.shape)

    img = G.synthesis(combined_dlatents, noise_mode="const", force_fp32=True)
    # img = G(combined_dlatents, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(save_path)


def main():

    feature_folder = "./features/"
    save_folder = "imgs"

    os.makedirs(save_folder, exist_ok=True)

    parser = argparse.ArgumentParser(description="Generate Audio Reactive Images")
    parser.add_argument("--model_path", type=str, help="Path to Pretrained StyleGAN")
    arguments = parser.parse_args()

    unit_length = 15

    high_pass_features = get_rmse_feature(feature_folder + "hpf_y_rmse.npy")
    band_pass_features = get_rmse_feature(feature_folder + "bpf_y_rmse.npy")
    low_pass_features = get_rmse_feature(feature_folder + "lpf_y_rmse.npy")

    diff_high_pass = high_pass_features[1:] - high_pass_features[:-1]
    diff_high_pass = diff_high_pass / np.amax(diff_high_pass, axis=0)

    diff_band_pass = band_pass_features[1:] - band_pass_features[:-1]
    diff_band_pass = diff_band_pass / np.amax(diff_band_pass, axis=0)

    diff_low_pass = low_pass_features[1:] - low_pass_features[:-1]
    diff_low_pass = diff_low_pass / np.amax(diff_low_pass, axis=0)

    print('Loading networks from "%s"...' % arguments.model_path)
    with dnnlib.util.open_url(arguments.model_path) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    seed = 1
    latents = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

    latents_high = latents
    latents_band = latents
    latents_low = latents

    length_high = 10
    length_mid = 3
    length_low = 10

    with open("./style.yaml") as file:
        documents = yaml.full_load(file)
        lpf = documents["lpf"]
        hpf = documents["hpf"]
        bpf = documents["bpf"]

    gen_image_mix(
        G,
        latents_high,
        latents_band,
        latents_low,
        lpf,
        hpf,
        bpf,
        save_path="./{}/{}.png".format(save_folder, 0),
    )
    for i in tqdm(range(diff_low_pass.shape[0])):
        latents_high = move_vec(latents_high.cpu(), length_high, diff_high_pass[i])
        latents_band = move_vec(latents_band.cpu(), length_mid, diff_band_pass[i])
        latents_low = move_vec(latents_low.cpu(), length_low, diff_low_pass[i])

        gen_image_mix(
            G,
            latents_high,
            latents_band,
            latents_low,
            lpf,
            hpf,
            bpf,
            save_path="./{}/{}.png".format(save_folder, i),
        )


if __name__ == "__main__":
    main()
