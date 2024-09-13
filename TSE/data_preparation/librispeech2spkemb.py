import argparse
import os
from shutil import copyfile
import numpy as np
np.random.seed(20030523)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--librispeech_root", type=str)
    parser.add_argument("--python_path", type=str)
    parser.add_argument("--remove_intermediates", type=str)
    args = parser.parse_args()
    root = args.root
    split = args.split
    librispeech_root = args.librispeech_root
    python_path = args.python_path
    assert args.remove_intermediates == "True" or args.remove_intermediates == "False"
    remove_intermediates = eval(args.remove_intermediates)

    wav_root = os.path.join(librispeech_root, split)
    temp_txt = os.path.join(root, "TSE", "temp_files", "temptxt.txt")
    os.makedirs(os.path.join(root, "TSE", "temp_files"), exist_ok=True)
    with open(temp_txt, "w") as w:
        for spk in os.listdir(wav_root):
            # We set-up that for training process, there is a multiple of enrollment speeches for one speaker
            # but for validation and test process, there is only an enrollment speech for one speaker
            if "train" in split:
                for chapter in os.listdir(os.path.join(wav_root, spk)):
                    for file in os.listdir(os.path.join(wav_root, spk, chapter)):
                        if not file.endswith(".txt"):
                            w.write(os.path.join(wav_root, spk, chapter, file) + "\n")
            else:
                wav_list = []
                for chapter in os.listdir(os.path.join(wav_root, spk)):
                    for file in os.listdir(os.path.join(wav_root, spk, chapter)):
                        if not file.endswith(".txt"):
                            wav_list.append(os.path.join(wav_root, spk, chapter, file))
                w.write(np.random.choice(wav_list, 1)[0] + "\n")
    os.system(
        f"{python_path} {root}/data_preparation/3D-Speaker/speakerlab/bin/infer_sv.py --model_id iic/speech_campplus_sv_zh-cn_16k-common --wavs {temp_txt}"
    )
    os.remove(temp_txt)
    tar_emb_root = os.path.join(
        root, "TSE", "data", "enroll_speech", "embeddings", "LibriSpeech_CAM", split
    )
    os.makedirs(tar_emb_root, exist_ok=True)
    for file in os.listdir(
        os.path.join(
            root,
            "TSE",
            "pretrained",
            "speech_campplus_sv_zh-cn_16k-common",
            "embeddings",
        )
    ):
        copyfile(
            os.path.join(
                os.path.join(
                    root,
                    "TSE",
                    "pretrained",
                    "speech_campplus_sv_zh-cn_16k-common",
                    "embeddings",
                    file,
                )
            ),
            os.path.join(tar_emb_root, file),
        )
        os.remove(
            os.path.join(
                os.path.join(
                    root,
                    "TSE",
                    "pretrained",
                    "speech_campplus_sv_zh-cn_16k-common",
                    "embeddings",
                    file,
                )
            )
        )

    if remove_intermediates == True:
        os.system(f"rm -r {root}/TSE/temp_files")
        os.system(f"rm -r {root}/TSE/pretrained")
