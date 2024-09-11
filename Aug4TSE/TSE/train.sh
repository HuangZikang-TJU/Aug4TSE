PYTHON=/Work21/2024/huangzikang/miniconda3/envs/new_tts/bin/python
EXP_NAME='Test'
OUTHOME=/Work21/2024/huangzikang/Github/Aug4TSE/TSE/out/${EXP_NAME}
# rm -rf ${OUTHOME}
mkdir -p ${OUTHOME}

PYTHONPATH=/Work21/2024/huangzikang/Github/Aug4TSE/TSE/ \
CUDA_VISIBLE_DEVICES=1 \
${PYTHON} \
    /Work21/2024/huangzikang/Github/Aug4TSE/TSE/train.py \
--cfg-path /Work21/2024/huangzikang/Github/Aug4TSE/TSE/configs/conf_dpccn_ERes2NetV2.yml \
2>&1 | tee ${OUTHOME}/train.out
