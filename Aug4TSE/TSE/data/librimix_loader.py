import os
import numpy as np
from pathlib import Path
import torch
import torchaudio
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from tqdm import tqdm

torch.random.manual_seed(42)
np.random.seed(42)


class Libri2Mix(Dataset):
    def __init__(
        self,
        paths,
        label_path,
        max_audio=None,
        subset="test",
        spk_emb_dir="",
        spk_dict=None,
    ):
        self.paths = paths
        self.label_path = label_path
        self.max_audio = max_audio
        self.subset = subset
        assert (self.subset != "test" and max_audio is not None) or (
            self.subset == "test" and max_audio is None
        )
        assert subset in ["train", "dev", "test"]
        self.spk_emb_dir = None
        if spk_dict is not None and len(spk_emb_dir) == 1:
            if subset == "train":
                self.spk_emb_dir = os.path.join(spk_emb_dir[0], "train-clean-100")
            elif subset == "dev":
                self.spk_emb_dir = os.path.join(spk_emb_dir[0], "dev-clean")
            else:
                self.spk_emb_dir = os.path.join(spk_emb_dir[0], "test-clean")
        self.spk_dict = spk_dict

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        wav, target_wavs, aud_name = self.__load_wavs(idx)
        spk_emb = self.__load_emb(aud_name)
        # mixture: L (train); 1 x L (test)
        # targets: 2 x L
        # spk_embeds: 2 x spk_emb_dim
        return {
            "mixture": wav,
            "targets": target_wavs,
            "spk_embeds": spk_emb,
            "aud_name": aud_name,
        }

    def __load_emb(self, aud_name):
        s1_name, s2_name = aud_name[:-4].split("_")
        s1_spk, _, _ = s1_name.split("-")
        s1_files = self.spk_dict[s1_spk]
        s1_emb_file = np.random.choice(s1_files)
        while s1_emb_file[:-4] == s1_name and len(s1_files) > 1:
            s1_emb_file = np.random.choice(s1_files)

        if self.spk_emb_dir is not None:
            s1_spk_emb = np.load(os.path.join(self.spk_emb_dir, s1_emb_file))
        else:
            s1_spk_emb = np.load(s1_emb_file)
        s1_spk_emb = torch.from_numpy(s1_spk_emb)

        s2_spk, _, _ = s2_name.split("-")
        s2_files = self.spk_dict[s2_spk]
        s2_emb_file = np.random.choice(s2_files)
        while s2_emb_file[:-4] == s2_name and len(s2_files) > 1:
            s2_emb_file = np.random.choice(s2_files)
        if self.spk_emb_dir is not None:
            s2_spk_emb = np.load(os.path.join(self.spk_emb_dir, s2_emb_file))
        else:
            s2_spk_emb = np.load(s2_emb_file)
        s2_spk_emb = torch.from_numpy(s2_spk_emb)

        spk_emb = torch.cat([s1_spk_emb, s2_spk_emb], dim=0)
        # spk_emb: 2 x spk_emb_dim
        return spk_emb

    def __load_wavs(self, idx):
        # Get mixture
        mixture_path = self.paths[idx]
        wav, sr = torchaudio.load(mixture_path)
        _, aud_name = os.path.split(mixture_path)

        # Get clean wavs
        clean_path = os.path.join(self.label_path, "s1", aud_name)
        wav_s1, sr = torchaudio.load(clean_path)
        clean_path = os.path.join(self.label_path, "s2", aud_name)
        wav_s2, sr = torchaudio.load(clean_path)

        if self.subset == "test":
            target_wavs = torch.concat([wav_s1, wav_s2], dim=0)
        else:
            max_audio_length = self.max_audio * sr
            if wav.shape[-1] < max_audio_length:  # pad audio
                diff = max_audio_length - wav.shape[-1]
                left_pad = diff // 2
                right_pad = diff - left_pad
                wav = torch.nn.functional.pad(wav, (left_pad, right_pad)).reshape(-1)
                wav_s1 = torch.nn.functional.pad(wav_s1, (left_pad, right_pad)).reshape(
                    1, -1
                )
                wav_s2 = torch.nn.functional.pad(wav_s2, (left_pad, right_pad)).reshape(
                    1, -1
                )
            else:  # crop audio
                start = torch.randint(0, (wav.shape[-1] - max_audio_length + 1), (1,))
                wav = wav[0, start : start + max_audio_length].reshape(-1)
                wav_s1 = wav_s1[0, start : start + max_audio_length].reshape(1, -1)
                wav_s2 = wav_s2[0, start : start + max_audio_length].reshape(1, -1)
            target_wavs = torch.concat([wav_s1, wav_s2], dim=0)
        # wav: L (train); 1 x L (test)
        # target_wavs: 2 x L
        return wav, target_wavs, aud_name


class Libri1Mix(Dataset):
    def __init__(
        self,
        paths,
        label_path,
        max_audio=None,
        subset="test",
        spk_emb_dir="",
        spk_dict=None,
    ):
        self.paths = paths
        self.label_path = label_path
        self.max_audio = max_audio
        self.subset = subset
        assert (self.subset != "test" and max_audio is not None) or (
            self.subset == "test" and max_audio is None
        )
        assert subset in ["train", "dev", "test"]
        if spk_dict is not None:
            if subset == "train":
                self.spk_emb_dir = os.path.join(spk_emb_dir, "train-clean-100")
            elif subset == "dev":
                self.spk_emb_dir = os.path.join(spk_emb_dir, "dev-clean")
            else:
                self.spk_emb_dir = os.path.join(spk_emb_dir, "test-clean")
        self.spk_dict = spk_dict
        assert False, "changes made on Libri2Mix but not Libri1Mix"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        wav, target_wav, aud_name = self.__load_wavs(idx)
        if self.spk_dict is not None:
            spk_emb = self.__load_emb(aud_name)
            wav = wav.reshape(1, -1)
            target_wav = target_wav.reshape(1, -1)
            return {"mixture": wav, "targets": target_wav, "spk_embeds": spk_emb}
        return wav, target_wav

    def __load_emb(self, aud_name):
        s1_name, s2_name = aud_name[:-4].split("_")
        s1_spk, _, _ = s1_name.split("-")
        s1_files = self.spk_dict[s1_spk]
        s1_emb_file = np.random.choice(s1_files)
        while s1_emb_file[:-4] == s1_name and len(s1_files) > 1:
            s1_emb_file = np.random.choice(s1_files)
        s1_spk_emb = np.load(os.path.join(self.spk_emb_dir, s1_emb_file))
        s1_spk_emb = torch.from_numpy(s1_spk_emb)

        return s1_spk_emb

    def __load_wavs(self, idx):
        # Get mixture
        mixture_path = self.paths[idx]
        wav, sr = torchaudio.load(mixture_path)
        _, aud_name = os.path.split(mixture_path)

        # Get clean wavs
        clean_path = os.path.join(self.label_path, "s1", aud_name)
        wav_s1, sr = torchaudio.load(clean_path)

        if self.subset == "test":  # if too long, crop and return segmented wavs
            return wav, wav_s1, aud_name
        else:
            max_audio_length = self.max_audio * sr
            if wav.shape[-1] < max_audio_length:  # pad audio
                diff = max_audio_length - wav.shape[-1]
                left_pad = diff // 2
                right_pad = diff - left_pad
                wav = torch.nn.functional.pad(wav, (left_pad, right_pad)).reshape(-1)
                wav_s1 = torch.nn.functional.pad(wav_s1, (left_pad, right_pad)).reshape(
                    -1
                )
            else:  # crop audio
                start = torch.randint(0, (wav.shape[1] - max_audio_length + 1), (1,))
                wav = wav[0, start : start + max_audio_length].reshape(-1)
                wav_s1 = wav_s1[0, start : start + max_audio_length]
        # wav: L (train); 1 x L (test)
        # target_wavs: 1 x L
        return wav, wav_s1, aud_name


def get_filepaths(audio_root, subset, mode="mix_clean", ftype="wav"):
    label_path = os.path.join(audio_root, subset)

    audio_dir = os.path.join(audio_root, subset, mode)
    print(f"Loading mixtures from {audio_dir}")
    file_paths = []
    for filename in os.listdir(audio_dir):
        if filename.endswith(ftype):
            filepath = os.path.join(audio_dir, filename)
            file_paths.append(filepath)
    return sorted(file_paths), label_path


def get_spk_dict(subset, spk_emb_dir_lst):
    print(
        f"Loading enrollment speeches' speaker embeddings from {spk_emb_dir_lst}/{subset}"
    )
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
            spk, _, _ = fname.split("-")
            if spk not in spk_dict:
                spk_dict[spk] = []
            spk_dict[spk].append(fname)

    return spk_dict


def get_dataloader(
    batch_size,
    num_workers,
    data_root="",
    spk_emb_dir=None,
    audio_length=4,
    sample_rate=16000,
    mode="mix_clean",
    train_subset="train-100",
    valid_subset="dev",
    **distributed_args,
):
    assert spk_emb_dir is not None, "data['spk_emb_dir'] not exist"
    spk_dict = get_spk_dict("train", spk_emb_dir)
    paths, label_path = get_filepaths(data_root, subset=train_subset, mode=mode)
    assert mode in ["mix_clean", "mix_both", "mix_single"]
    if mode in ["mix_clean", "mix_both"]:
        train_set = Libri2Mix(
            paths,
            label_path,
            max_audio=audio_length,
            subset="train",
            spk_emb_dir=spk_emb_dir,
            spk_dict=spk_dict,
        )
    else:
        train_set = Libri1Mix(
            paths,
            label_path,
            max_audio=audio_length,
            subset="train",
            spk_emb_dir=spk_emb_dir,
            spk_dict=spk_dict,
        )

    sampler = (
        DistributedSampler(
            train_set,
            shuffle=True,
        )
        if distributed_args["distributed"]
        else None
    )

    train_loader = DataLoader(
        train_set,
        shuffle=(sampler is None),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        sampler=sampler,
        collate_fn=collate_fn,
    )

    paths, label_path = get_filepaths(data_root, subset=valid_subset, mode=mode)
    if spk_emb_dir is not None:
        spk_dict = get_spk_dict("dev", spk_emb_dir)
    else:
        spk_dict = None
    if mode in ["mix_clean", "mix_both"]:
        dev_set = Libri2Mix(
            paths,
            label_path,
            max_audio=audio_length,
            subset="dev",
            spk_emb_dir=spk_emb_dir,
            spk_dict=spk_dict,
        )
    else:
        dev_set = Libri1Mix(
            paths,
            label_path,
            max_audio=audio_length,
            subset="dev",
            spk_emb_dir=spk_emb_dir,
            spk_dict=spk_dict,
        )
    sampler = (
        DistributedSampler(dev_set, shuffle=False)
        if distributed_args["distributed"]
        else None
    )
    dev_loader = DataLoader(
        dev_set,
        shuffle=False,
        batch_size=max(batch_size, 8),
        num_workers=num_workers,
        # drop_last=False,
        drop_last=True,
        sampler=sampler,
        collate_fn=collate_fn,
    )
    return train_loader, dev_loader


def collate_fn(samples):
    target_wavs = [s["targets"] for s in samples]
    spk_embeds = [s["spk_embeds"] for s in samples]
    if target_wavs[0].shape[0] > 1:  # multiple speakers
        wavs = [torch.stack([s["mixture"], s["mixture"]], dim=0) for s in samples]
    else:
        wavs = [s["mixture"] for s in samples]
    wavs = torch.cat(wavs)
    target_wavs = torch.cat(target_wavs)
    spk_embeds = torch.cat(spk_embeds)
    # if len(samples) == 1:
    #     return wavs, target_wavs, spk_embeds, samples[0]["aud_name"]

    # wavs: (B x 2) x L (train); (B x 2) x L (test)
    # targets: (B x 2) x L
    # spk_embeds: (B x 2) x spk_emb_dim
    return wavs, target_wavs, spk_embeds
