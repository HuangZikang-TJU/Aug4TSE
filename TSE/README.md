# Train a new TSE model
We will use the TSE model DPCCN and the pretrained speaker encoder CAM++ from the [3D-Speaker ToolKit](https://github.com/modelscope/3D-Speaker) to conduct the following example.

### Stage 1: Prepare Training Data
Before executing this phase, we assume that you have the Libri2Mix and LibriSpeech datasets stored locally or accessible.

- Convert speeches in LibriSpeech into speaker embeddings. The speaker embeddings will be used during the training and validation process. You need to change the variables in get_spkemb4training.sh
```
cd TSE
bash get_spkemb4training.sh
```
- At the end of this stage, the directory structure of 'data' should look like this:
```
Aug4TSE/TSE/data/enroll_speech/embeddings/LibriSpeech_CAM
train-clean-100:
 - 19-198-0000.npy
 - 19-198-0001.npy
 - 19-198-0000.npy
 - ...
dev-clean:
 - 84-121123-0024.npy
 - 174-50561-0012.npy
 - 251-136532-0014.npy
 - ...
```
- As the version of 3D-Speaker ToolKit may change, we recommend checking whether the .npy files in /Aug4TSE/TSE/data/enrollment_speech have a shape of (C,) rather than (1, C).
### Stage 2: Train TSE model with Specific Config
- You need to change the variables in train.sh and choose a specific config. If this is your first time running this code, you can use the default config.
- Change the variables in the config such as "data_root" and "spk_emb_dir". 
- Please ensure that you run the following command at the working path Aug4TSE/TSE
```
bash train.sh
```
- You will find the checkpoint in Aug4TSE/TSE/exp, and you can choose the checkpoint that has the best performance in validation by viewing the log file in Aug4TSE/TSE/out.

- If you don't have time to train a new model, we provide a model with the exact same pipeline and store it at path Aug4TSE/TSE/ckpt_DPCCN_CAM.ckpt using a GPU RTX4090D and CUDA version 12.4. You can use this checkpoint to conduct the evaluation process.

# Evaluate a TSE model
#### Now, you have a checkpoint that includes the TSE model. You can evaluate it with different enrollment speeches.
- We recommend that you first view /Aug4TSE/data_preparation before proceeding with the content below.
- First, you need to convert the enrollment speeches into speaker embeddings following the pipeline in /Aug4TSE/data_preparation. The prepared speaker embeddings should look like this:
```
Aug4TSE/data_preparation/data/available_embedding/{0.5s, 0.5s_TTS_concat, other you want}/test-clean
 - 61-70968-0041.npy
 - 121-121726-0010.npy
 - 237-126133-0000.npy
 - ...
```
- Then, you should change the basic variables in eval.sh such as "cfg-path" and "model-path".
- Importantly, you should change the setting "spk_emb_dir" in the config file that you choose in eval.sh to determine which enrollment speeches' speaker embeddings you want to use for evaluation.

#### Here are the evaluation results obtained by running the entire code.
| Enroll. | SDRi | SI-SDRi |
|:---------:|:---------:|:---------:|
| 0.5s | 7.01 | 5.47 |
| 0.5s_TTS_concat | 8.26 | 6.86 |
