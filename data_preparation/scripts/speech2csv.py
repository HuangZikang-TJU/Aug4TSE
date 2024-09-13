import argparse
import os
from shutil import copyfile
import whisper
import tqdm
import numpy as np

np.random.seed(20230523)


def norm(text: str):
    text = text.lower()
    text = text.replace(".", "")
    return text


target_text_list = []


def get_target_texts(root: str):
    with open(os.path.join(root, "data", "target_texts.txt"), "r") as r:
        for line in r.readlines():
            target_text_list.append(line.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--split", type=str)
    args = parser.parse_args()
    root = args.root
    split = args.split

    # create csv
    os.makedirs(f"{root}/data/available_csv", exist_ok=True)
    csv_path = f"{root}/data/available_csv/{split}.csv"
    wav_root = f"{root}/data/available_speech/{split}"

    get_target_texts(root)
    model = whisper.load_model("small").cuda()
    with open(csv_path, "w") as w:
        for spk in tqdm.tqdm(os.listdir(wav_root)):
            for file in tqdm.tqdm(os.listdir(os.path.join(wav_root, spk))):
                result = model.transcribe(os.path.join(wav_root, spk, file))["text"]
                result = norm(result)
                target_text = np.random.choice(target_text_list, 1)[0]
                w.write(
                    os.path.join(wav_root, spk, file)
                    + "\t"
                    + result
                    + "\t"
                    + "invalid_padding"
                    + "\t"
                    + target_text
                    + "\n"
                )
