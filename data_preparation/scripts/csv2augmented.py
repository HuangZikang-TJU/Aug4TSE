import argparse
import os
from shutil import copyfile
import torchaudio
import torch
import torch.nn.functional as F
import numpy as np
import os
import tqdm
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import librosa

def download_TTS(root: str):
    from modelscope import snapshot_download

    snapshot_download(
        "damo/speech_synthesizer-laura-en-libritts-16k-codec_nq2-pytorch",
        cache_dir=f"{root}/TTS_models",
    )


def synthesis(root: str, split: str, sample_time: int):
    my_pipeline = pipeline(
        task=Tasks.text_to_speech,
        model=f"{root}/TTS_models/damo/speech_synthesizer-laura-en-libritts-16k-codec_nq2-pytorch",
    )
    tsv = f"{root}/data/available_csv/{split}.csv"
    results_dir = f"{root}/temp_files/TTS/{split}"
    with open(tsv, "r") as r:
        for id, line in tqdm.tqdm(enumerate(r.readlines())):
            ori_wav, ori_text, tar_wav, tar_text = line.strip().split("\t")
            for i in range(sample_time):
                output_wav = my_pipeline(tar_text, ori_text, ori_wav)["output_wav"]
                output_wav = output_wav / np.max(output_wav)
                os.makedirs(os.path.join(results_dir, f"sample_{i}"), exist_ok=True)
                write_wav_path = os.path.join(
                    results_dir,
                    f"sample_{i}",
                    f"{ori_wav.split('/')[-1].split('.')[0]}_TTS.wav",
                )
                output_wav = torch.from_numpy(output_wav)
                torchaudio.save(write_wav_path, output_wav, 16000)


def choose_synthesis(root: str, split: str, sample_times: int, python_path: str, remove_intermediates: bool):
    for id in range(sample_times):
        ttswav_root = f"{root}/temp_files/TTS/{split}/sample_{str(id)}"
        des = f"{root}/temp_files/TTS/{split}/sample_{str(id)}.txt"
        with open(des, "w") as f:
            for file in os.listdir(ttswav_root):
                f.write(os.path.join(ttswav_root, file) + "\n")
        os.system(
            f"{python_path} {root}/3D-Speaker/speakerlab/bin/infer_sv.py --model_id iic/speech_campplus_sv_zh-cn_16k-common --wavs {des}"
        )
        os.remove(des)

        evaltext = f"{root}/temp_files/TTS/{split}/sample_{str(id)}_eval.txt"
        ref_emb_path = f"{root}/data/available_embedding/{split}"
        src_emb_path = (
            f"{root}/pretrained/speech_campplus_sv_zh-cn_16k-common/embeddings"
        )

        with open(evaltext, "w") as w:
            for file in os.listdir(src_emb_path):
                ref_path = os.path.join(
                    ref_emb_path, f"{file.split('.npy')[0].split('_')[0]}.npy"
                )
                src_path = os.path.join(src_emb_path, file)
                if (
                    os.path.exists(src_path) == False
                    or os.path.exists(ref_path) == False
                ):
                    w.write(ref_path + "|" + src_path + "\t" + str(0.0) + "\n")
                    continue
                src_emb = np.load(src_path)
                ref_emb = np.load(ref_path)
                src_emb = torch.from_numpy(src_emb)
                ref_emb = torch.from_numpy(ref_emb)
                sim = F.cosine_similarity(src_emb, ref_emb, axis=0)
                w.write(ref_path + "|" + src_path + "\t" + str(sim.item()) + "\n")

    ttswav_root = f"{root}/temp_files/TTS/{split}"
    max_sim = []
    max_idx = []
    for i in range(sample_times):
        eval_txt = os.path.join(ttswav_root, f"sample_{str(i)}_eval.txt")
        with open(eval_txt, "r") as r:
            num = 0
            for line in r.readlines():
                line = line.strip()
                if line.split("\t")[1] != 'nan':
                    grade = eval(line.split("\t")[1])
                else:
                    grade = 0.0
                
                if i == 0:
                    max_sim.append(grade)
                    max_idx.append(i)
                else:
                    if grade > max_sim[num]:
                        max_sim[num] = grade
                        max_idx[num] = i
                num += 1
    eval_txt = f"{root}/temp_files/TTS/{split}/best_eval.txt"
    os.makedirs(f"{root}/temp_files/TTS/{split}/choose", exist_ok=True)

    with open(eval_txt, "w") as w:
        with open(os.path.join(ttswav_root, f"sample_{str(i)}_eval.txt"), 'r') as r:
            for i,line in enumerate(r.readlines()):
                line = line.strip()
                file = line.split('\t')[0].split('/')[-1].split('.npy')[0]
                src_path = f"{root}/temp_files/TTS/{split}/sample_{str(max_idx[i])}/{file}.wav"
                tar_path = f"{root}/temp_files/TTS/{split}/choose/{file}.wav"
                copyfile(src_path, tar_path)
                w.write(
                    f"max_idx:{str(max_idx[i])}"
                    + "\t"
                    + f"max_sim:{str(max_sim[i])}"
                    + "\n"
                )

    os.system(f"rm -r {root}/pretrained/speech_campplus_sv_zh-cn_16k-common/embeddings")
    for i in range(sample_times):
        os.remove(os.path.join(ttswav_root, f"sample_{str(i)}_eval.txt"))
    if remove_intermediates == True:
        os.remove(eval_txt)
        for i in range(sample_times):
            os.system(f"rm -r {root}/temp_files/TTS/{split}/sample_{str(i)}")

def concat(root: str, split: str, remove_intermediates: bool):
    concat_tar_path = f"{root}/data/available_speech/{split}_TTS_concat"
    os.makedirs(concat_tar_path, exist_ok=True)
    src_path = f"{root}/data/available_speech/{split}"
    TTS_path = f"{root}/temp_files/TTS/{split}/choose"
    for spk in os.listdir(src_path):
        os.makedirs(os.path.join(concat_tar_path,spk), exist_ok=True)
        for file in os.listdir(os.path.join(src_path,spk)):
            src_wav, _ = librosa.load(os.path.join(src_path,spk,file), sr=16000)
            TTS_wav, _ = librosa.load(os.path.join(TTS_path,f"{file.split('.')[0]}_TTS.wav"), sr=16000)
            sf.write(os.path.join(concat_tar_path,spk,file), np.concatenate([src_wav, TTS_wav],axis=0), 16000)
    if remove_intermediates:
        os.system(f"rm -r {root}/temp_files/TTS")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--python_path", type=str)
    parser.add_argument("--sample_times", type=int)
    parser.add_argument("--remove_intermediates", type=str)
    args = parser.parse_args()
    root = args.root
    split = args.split
    python_path = args.python_path
    sample_times = args.sample_times
    assert args.remove_intermediates == "True" or args.remove_intermediates == "False"
    remove_intermediates = eval(args.remove_intermediates)

    download_TTS(root)
    # synthesize {sample_times} tiems
    synthesis(root, split, sample_times)

    choose_synthesis(root, split, sample_times, python_path, remove_intermediates)

    concat(root, split, remove_intermediates)
