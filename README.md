# Augmenting Short Enrollment Speech via Synthesis for Target Speaker Extraction
> A open-source code for ICASSP2025

> In this project, we will introduce: 1) How to train the Target Speaker Extraction (TSE) models in our method. 2) How to use Text-to-Speech(TTS) system to conduct short enrollment speech augmentation which is proposed in our paper. 3) How to evaluate the TSE models using enrollment speeches.

# Train a new TSE model
Follow the steps in ./TSE to create:
- {train,valid}.npy enrollment speeches' speaker embeddings
- _ckpt_epoch_xxx.ckpt a TSE model's checkpoint

# Augmenting based on the given short enrollment speeches
Follow the steps in ./data_preparation to create:
- {augmented_speech}.wav augmented speeches based on short enrollment speeches
- {augmented_speech}.npy speaker embedding of augmented speeches

# Evaluate the TSE models using enrollment speeches
Follow the steps in ./data_preparation to create:
- {test}.npy the speaker embeddings of enrollment speeches that you want to use to evaluate TSE models

Follow the steps in ./TSE to create:
- evalutation.log the evaluation results