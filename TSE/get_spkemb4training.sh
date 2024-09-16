PYTHON=/Work21/2024/huangzikang/miniconda3/envs/Aug4TSE/bin/python # python.exe path 
remove_intermediates="False" # False or True # the whole process will generate many intermediates. If you first conduct this experiment, we recommend "False"

root=/Work21/2024/huangzikang/Github/Aug4TSE #/path/to/Aug4TSE
LibriSpeech_root=/CDShare3/LibriSpeech #./LibriSpeech

GPU=0 #if the process need a GPU, your choosen GPU will be used.

for split in dev-clean train-clean-100
do
echo "convert subset ${split} of librispeech into enrollment speeches' speaker embedding"
    PYTHONPATH=${root} \
    CUDA_VISIBLE_DEVICES=${GPU} \
    ${PYTHON} ${root}/TSE/data_preparation/librispeech2spkemb.py \
        --root ${root} \
        --split ${split} \
        --librispeech_root ${LibriSpeech_root} \
        --python_path ${PYTHON} \
        --remove_intermediates ${remove_intermediates}
done
wait