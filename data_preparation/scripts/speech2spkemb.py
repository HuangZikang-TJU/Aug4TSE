import argparse
import os
from shutil import copyfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--python_path", type=str)
    parser.add_argument("--remove_intermediates", type=str)
    args = parser.parse_args()
    root = args.root
    split = args.split
    python_path = args.python_path
    assert args.remove_intermediates == "True" or args.remove_intermediates == "False"
    remove_intermediates = eval(args.remove_intermediates)

    os.makedirs(os.path.join(root, "temp_files"), exist_ok=True)
    temp_txt = os.path.join(root, "temp_files", "original_wav_2_spkemb.txt")
    wav_root = os.path.join(root, "data", "available_speech", split)
    with open(temp_txt, "w") as w:
        for spk in os.listdir(wav_root):
            for file in os.listdir(os.path.join(wav_root, spk)):
                w.write(os.path.join(wav_root, spk, file) + "\n")
    # Use CAM++ for speaker embeddings. If you want to use other speaker encoder, you can replace model_id.
    os.system(
        f"{python_path} {root}/3D-Speaker/speakerlab/bin/infer_sv.py --model_id iic/speech_campplus_sv_zh-cn_16k-common --wavs {temp_txt}"
    )

    src_emb_root = f"{root}/pretrained/speech_campplus_sv_zh-cn_16k-common/embeddings"
    tar_emb_root = f"{root}/data/available_embedding/{split}/test-clean"
    os.makedirs(tar_emb_root, exist_ok=True)
    for file in os.listdir(src_emb_root):
        copyfile(os.path.join(src_emb_root, file), os.path.join(tar_emb_root, file))

    os.system(f"rm -r {root}/pretrained/speech_campplus_sv_zh-cn_16k-common/embeddings")
    os.remove(temp_txt)
    if remove_intermediates:
        os.system(f"rm -r {root}/pretrained")
        os.system(f"rm -r {root}/temp_files")
