# Train a new TSE model
We will use TSE model DPCCN and pretrained speaker encoder CAM++ that is from [3D-Speaker ToolKit](https://github.com/modelscope/3D-Speaker) to conduct the following example.

### Stage 1: Prepare Training Data
Prior to executing this phase, we assume that you have locaaly stored or can access the Libri2Mix and LibriSpeech dataset.

- Invert speeches in LibriSpeech into speaker embeddings. The speaker embeddings will be used during training and validation process. You need to change the variables in get_spkemb4training.sh  
```
cd TSE
bash get_spkemb4training.sh
```

### Stage 2: Train TSE model with Specific Config
- You need to change the variables in train.sh and choose a specific config. If you first run this code, you can use the default config.
- Change the variables in config such as "data_root" and "spk_emb_dir". 
- Please ensure that you run the following command at work-path Aug4TSE/TSE
```
bash train.sh
```
- You will get the checkpoint in Aug4TSE/TSE/exp, and you can choose the checkpoint which has the best performance in validation by seeing the log file in Aug4TSE/TSE/out.