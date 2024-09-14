PYTHON=/Work21/2024/huangzikang/miniconda3/envs/new_tts/bin/python # python.exe path 
remove_intermediates="False" # False or True # the whole process will generate many intermediates. If you first conduct this experiment, we recommend "False"

root=/Work21/2024/huangzikang/Github/Aug4TSE/data_preparation #/path/to/data_preparation
split=0.5s #short enrollment speech.  ./data_preparation/data/available_speech/split

GPU=0 #if the process need a GPU, your choosen GPU will be used.

#### for stage 3, the sample_times refers the times that TTS model generate for one short enrollment speech
#### if you first conduct this experiment, we recommend a low number
sample_times=2

stage=4
stop_stage=4

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "convert short enrollment speech into speaker embeddings"
    PYTHONPATH=${root} \
    CUDA_VISIBLE_DEVICES=${GPU} \
    ${PYTHON} ${root}/scripts/speech2spkemb.py \
        --root ${root} \
        --split ${split} \
        --python_path ${PYTHON} \
        --remove_intermediates ${remove_intermediates}
fi

# if the text of short enrollment speech is applied, please add it by hand and skip this process.
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "generate prerequisites csv, the setting about synthesis of augmented enrollment speech"
    PYTHONPATH=${root} \
    CUDA_VISIBLE_DEVICES=${GPU} \
    ${PYTHON} ${root}/scripts/speech2csv.py \
        --root ${root} \
        --split ${split}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "synthesize speeches based on the prerequisites csv."
    PYTHONPATH=${root} \
    CUDA_VISIBLE_DEVICES=${GPU} \
    ${PYTHON} ${root}/scripts/csv2augmented.py \
        --root ${root} \
        --split ${split} \
        --python_path ${PYTHON} \
        --sample_times ${sample_times} \
        --remove_intermediates ${remove_intermediates}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "convert augmented enrollment speech into speaker embeddings"
    PYTHONPATH=${root} \
    ${PYTHON} ${root}/scripts/speech2spkemb.py \
        --root ${root} \
        --split ${split}_TTS_concat \
        --python_path ${PYTHON} \
        --remove_intermediates ${remove_intermediates}
fi