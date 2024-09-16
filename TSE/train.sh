PYTHON=/Work21/2024/huangzikang/miniconda3/envs/Aug4TSE/bin/python #python.exe path
EXP_NAME='dpccn_Test' #the dirname, and the output log will be saved in Aug4TSE/TSE/out/{EXP_NAME}
root=/Work21/2024/huangzikang/Github/Aug4TSE # /path/to/Aug4TSE
OUTHOME=${root}/TSE/out/${EXP_NAME}
# rm -rf ${OUTHOME}
mkdir -p ${OUTHOME}
GPU=2 # the GPU that will be used to train TSE model. You can also conduct distributed training, and remember to change the related setting in config.

#cfg-path is the important setting, please check it carefully.
PYTHONPATH=${root}/TSE/ \
CUDA_VISIBLE_DEVICES=${GPU} \
${PYTHON} \
    ${root}/TSE/train.py --cfgpath ${root}/TSE/configs/conf_dpccn_CAM.yml \
2>&1 | tee ${OUTHOME}/train.out
