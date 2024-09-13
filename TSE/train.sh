PYTHON=/Work21/2024/huangzikang/miniconda3/envs/new_tts/bin/python #python.exe path
EXP_NAME='dpccn_Test' #the dirname, and the output log will be saved in Aug4TSE/TSE/out/{EXP_NAME}
root=/Work21/2024/huangzikang/Github/Aug4TSE # /path/to/Aug4TSE
OUTHOME=${root}/TSE/out/${EXP_NAME}
# rm -rf ${OUTHOME}
mkdir -p ${OUTHOME}
GPU=0 # the GPU that will be used to train TSE model. You can also conduct distributed training, and remember to change the related setting in config.

PYTHONPATH=${root}/TSE/ \
CUDA_VISIBLE_DEVICES=${GPU} \
${PYTHON} \
    ${root}/TSE/train.py \
    --cfg-path ${root}/TSE/configs/conf_dpccn_CAM.yml \
2>&1 | tee ${OUTHOME}/train.out
