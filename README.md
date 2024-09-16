# Augmenting Short Enrollment Speech via Synthesis for Target Speaker Extraction
> Open-source code for ICASSP 2025

> In this project, we will introduce: 1) How to train the Target Speaker Extraction (TSE) models using our method. 2) How to use the Text-to-Speech (TTS) system to conduct short enrollment speech augmentation, as proposed in our paper. 3) How to evaluate the TSE models using enrollment speeches.

Prepare the environment
```
cd Aug4TSE
conda create --name Aug4TSE python=3.8.18
conda activate Aug4TSE
pip install -r requirements.txt
```

# Train a new TSE model
Follow the steps in ./TSE to create:
- {train,valid}.npy: enrollment speeches' speaker embeddings
- _ckpt_epoch_xxx.ckpt: a TSE model's checkpoint

# Augmenting based on the given short enrollment speeches
Follow the steps in ./data_preparation to create:
- {augmented_speech}.wav: augmented speeches based on short enrollment speeches
- {augmented_speech}.npy: speaker embeddings of augmented speeches

# Evaluate the TSE models using enrollment speeches
Follow the steps in ./data_preparation to create:
- {test}.npy: the speaker embeddings of enrollment speeches that you want to use to evaluate TSE models

Follow the steps in ./TSE to create:
- evaluation.log: the evaluation results