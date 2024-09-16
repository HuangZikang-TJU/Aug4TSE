PYTHON=/Work21/2024/huangzikang/miniconda3/envs/Aug4TSE/bin/python #python.exe path
EXP_NAME='dpccn_Test' #the dirname, and the evaluation log will be saved in Aug4TSE/TSE/out/{EXP_NAME}
root=/Work21/2024/huangzikang/Github/Aug4TSE # /path/to/Aug4TSE
OUTHOME=${root}/TSE/out/${EXP_NAME}
# rm -rf ${OUTHOME}
mkdir -p ${OUTHOME}
GPU=0 # the GPU that will be used to evaluate TSE model.

#cfg-path is the important setting, please check it carefully.
PYTHONPATH=${root}/TSE/ \
CUDA_VISIBLE_DEVICES=${GPU} \
${PYTHON} \
    ${root}/TSE/eval_librimix.py \
        --model-path ${root}/TSE/ckpt_DPCCN_CAM.ckpt \
        --cfg-path ${root}/TSE/configs/conf_dpccn_CAM.yml \
2>&1 | tee ${OUTHOME}/evaluation.out