CUDA_IDX=0
PYTHON=/Work21/2024/huangzikang/miniconda3/envs/new_tts/bin/python

EXP_NAME='Test'

OUTHOME=/Work21/2024/huangzikang/Github/Aug4TSE/TSE/out/${EXP_NAME}
# rm -rf ${OUTHOME}
mkdir -p ${OUTHOME}

PYTHONPATH=/Work21/2024/huangzikang/Github/Aug4TSE/TSE \
CUDA_VISIBLE_DEVICES=${CUDA_IDX} \
${PYTHON} /Work21/2024/huangzikang/Github/Aug4TSE/TSE/eval_librimix.py \
--model-path /Work21/2024/huangzikang/Github/Aug4TSE/TSE/exp/dpccn_expand_mixtureonly/_ckpt_epoch_1.ckpt \
--cfg-path /Work21/2024/huangzikang/Github/Aug4TSE/TSE/configs/conf_dpccn_ERes2NetV2.yml \
2>&1 | tee ${OUTHOME}/eval.out