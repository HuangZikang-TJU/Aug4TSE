# Augmenting Short Enrollment Speech via Synthesis for Target Speaker Extraction
> A open-source code for ICASSP2025

> In this project, we will introduce: 1) How to train the Target Speaker Extraction (TSE) models in our method. 2) How to use Text-to-Speech(TTS) system to conduct short enrollment speech augmentation which is proposed in our paper. 3) How to evaluate the TSE models using enrollment speeches.

# Train a new model
Follow the steps in ./TSE to create:
- {train,valid}.npy enrollment speeches' speaker embeddings
- _ckpt_epoch_xxx.ckpt a TSE model's checkpoint

