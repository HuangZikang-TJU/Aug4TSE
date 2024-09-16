import argparse
import os
from torchmetrics.functional import (
    scale_invariant_signal_distortion_ratio,
    signal_distortion_ratio,
)
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import soundfile as sf
import sys
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models.bsrnn.bsrnn import BSRNN
from models.dpccn.dpccn import DPCCN

from losses import singlesrc_neg_sisdr
from data.librimix_loader import Libri2Mix, collate_fn

model_dict = {"dpccn": DPCCN, "bsrnn": BSRNN}
device = "cuda:0"


def load_best_param(model_path, model, gpu=False):
    if not os.path.exists(model_path):
        exit("log path error:" + model_path)
    ckpt = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except Exception as e:
        new_state_dict = {}
        for key, val in ckpt["model"].items():
            if key.startswith("model"):
                new_state_dict[key[6:]] = val
        model.load_state_dict(new_state_dict, strict=True)
    print("load param ok", model_path)
    # print("valid_losses:", ckpt["valid_losses"])
    if gpu:
        model.cuda(device)
    return model.eval()


def get_filepaths(audio_root, subset, mode="mix_clean", ftype="wav"):
    label_path = os.path.join(audio_root, subset)

    audio_dir = os.path.join(audio_root, subset, mode)
    print(f"Loading from {audio_dir}")
    file_paths = []
    for filename in os.listdir(audio_dir):
        if filename.endswith(ftype):
            filepath = os.path.join(audio_dir, filename)
            file_paths.append(filepath)
    return sorted(file_paths), label_path


def get_spk_dict(subset, spk_emb_dir_lst):
    print(f"Loading from {spk_emb_dir_lst}/{subset}")
    assert subset in ["train", "dev", "test"]
    spk_dict = {}
    for spk_emb_dir in spk_emb_dir_lst:
        if subset == "train":
            spk_emb_dir = os.path.join(spk_emb_dir, "train-clean-100")
        elif subset == "dev":
            spk_emb_dir = os.path.join(spk_emb_dir, "dev-clean")
        else:
            spk_emb_dir = os.path.join(spk_emb_dir, "test-clean")
        for fname in os.listdir(spk_emb_dir):
            if fname.endswith("gen.npy"):
                # if fname.endswith('gen_only_lm.npy'):
                continue
            spk, _, _ = fname.split("-")
            if spk not in spk_dict:
                spk_dict[spk] = []
            if len(spk_emb_dir_lst) > 1:
                spk_dict[spk].append(os.path.join(spk_emb_dir, fname))
            else:
                spk_dict[spk].append(fname)
    return spk_dict


def get_dataloader(
    data_root, batch_size=1, num_workers=4, spk_emb_dir=None, mode=None, subset="test"
):
    if spk_emb_dir is not None:
        spk_dict = get_spk_dict("test", spk_emb_dir)
    else:
        spk_dict = None
    paths, label_path = get_filepaths(data_root, subset=subset, mode=mode)
    assert mode in ["mix_clean", "mix_both", "mix_single"]
    if mode in ["mix_clean", "mix_both"]:
        test_set = Libri2Mix(
            paths, label_path, subset="test", spk_emb_dir=spk_emb_dir, spk_dict=spk_dict
        )
    else:
        test_set = Libri1Mix(
            paths, label_path, subset="test", spk_emb_dir=spk_emb_dir, spk_dict=spk_dict
        )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn if spk_dict is not None else None,
    )

    return test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--cfg-path", type=str)
    args = parser.parse_args()

    # load model
    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)
    print(cfg)
    model = model_dict[cfg["training"]["model_name"]](**cfg["model"])
    # print(model)
    model = load_best_param(args.model_path, model, gpu=True)

    # get dataloader
    test_loader = get_dataloader(
        data_root=cfg["data"]["data_root"],
        batch_size=1,
        num_workers=1,
        spk_emb_dir=cfg["data"]["spk_emb_dir"],
        mode=cfg["data"]["mode"],
    )
    print("len_test_loader:", len(test_loader))
    with torch.no_grad():
        sum_sisdr = 0
        sum_sdr = 0
        sum_ori_sisdr = 0
        sum_ori_sdr = 0

        for k, (mixture_wav, target_wavs, spk_embeds) in tqdm.tqdm(
            enumerate(test_loader)
        ):
            mixture_wav = mixture_wav.squeeze(1)
            mixture_wav = mixture_wav.to(device)
            target_wavs = target_wavs.to(device)
            spk_embeds = spk_embeds.to(device)

            estimate = model(mixture_wav, spk_embeds)

            original_si_sdr = scale_invariant_signal_distortion_ratio(
                mixture_wav, target_wavs
            )
            original_sdr = signal_distortion_ratio(mixture_wav, target_wavs)
            si_sdr = scale_invariant_signal_distortion_ratio(estimate, target_wavs)
            sdr = signal_distortion_ratio(estimate, target_wavs)

            ori_si_sdr = original_si_sdr.mean().item()
            ori_sdr = original_sdr.mean().item()
            si_sdr = si_sdr.mean().item()
            sdr = sdr.mean().item()

            sum_ori_sisdr += ori_si_sdr
            sum_ori_sdr += ori_sdr
            sum_sisdr += si_sdr
            sum_sdr += sdr
        print(
            "extraced_sisdr:",
            sum_sisdr / len(test_loader),
            "original_sisdr:",
            sum_ori_sisdr / len(test_loader),
            "sisdr_improvement:",
            (sum_sisdr - sum_ori_sisdr) / len(test_loader),
        )
        print(
            "extraced_sdr:",
            sum_sdr / len(test_loader),
            "original_sdr:",
            sum_ori_sdr / len(test_loader),
            "sdr_improvement:",
            (sum_sdr - sum_ori_sdr) / len(test_loader),
        )
