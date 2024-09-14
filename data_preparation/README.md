# Augmenting based on the given short enrollment speeches
Prior to executing this phase, we assume that you have locaaly stored or can access the short enrollment speeches. If you don't have them, we also provide some examples at Aug4TSE/data_preparation/data/available_speech/0.5s , you can use these examples to conduct the following pipeline.

If you just want to evaluate the TSE models using enrollment speeches, you can only see stage 1. If you see this introduction for the first time, we still recommend you read the all content.

We will use pretrained TTS system [LauraTTS](https://github.com/modelscope/FunCodec), pretrained speaker encoder CAM++ that is from [3D-Speaker ToolKit](https://github.com/modelscope/3D-Speaker) and ASR model Whisper-small that is from [OpenAI](https://github.com/openai/whisper) generate augmented enrollment speeches and corresponding speaker embeddings.

There are 4 stages in get_augmented.sh, if you first follow the pipeline, we recommend you run the stage one by one. You need to change the basic variables in get_augmented.sh
```
cd data_preparation
bash get_augmented.sh
```

### Stage 1: Convert Short Enrollment Speech into Speaker Embeddings
```
stage=1
stop_stage=1
```
In this stage, the short enrollment speeches will be embedded into speaker embeddings by the pretrained speaker encoder.
- At the end of this stage, the directory structure of 'data' should look like this:
```
Aug4TSE/data_preration/data/available_embedding/0.5s/test-clean
 - 61-70968-0041.npy
 - 121-121726-0010.npy
 - 237-126133-0000.npy
 - ...
```

### Stage 2: Prepare augmenting csv file based on short enrollment speeches
```
stage=2
stop_stage=2
```
In this stage, we will get a csv file including source_wav, source_text, _, target_text. It is a prerequisite for the following augmenting.
- At the end of this stage, the directory structure of 'data' should look like this:
```
Aug4TSE/data_preration/data/available_csv/0.5s.csv
 - /Work21/2024/huangzikang/Github/Aug4TSE/data_preparation/data/available_speech/0.5s/61/61-70968-0041.wav	 i like it	invalid_padding	whom the sister of the khan of the khazars had borne to him during his exile so ended the house of heraclius after it had sat for five generations and one hundred and one years on the throne of constantinople
 - ...
```

### Stage 3: Synthesize Speeches Based On The Prerequisite CSV
```
stage=3
stop_stage=3
```
In this stage, we will get the augmented enrollment speech, which is a concatenation speech of original short enrollment speech and synthesized speech based on the prerequisite csv file obtained in the stage 2. 
- At the end of this stage, the directory structure of 'data' should look like this:
```
Aug4TSE/data_preration/data/available_speech/0.5s_TTS_concat
 - 61/61-70968-0041.wav
 - 121/121-121726-0010.wav
 - 237/237-126133-0000.wav
 - ...
```

### Stage 4: Convert Augmented Enrollment Speeches Into Speaker Embeddings
```
stage=4
stop_stage=4
```
In this stage, the augmented enrollment speeches will be embedded into speaker embeddings by the pretrained speaker encoder. You can use these speaker embeddings to evaluate the TSE models.
- At the end of this stage, the directory structure of 'data' should look like this:
```
Aug4TSE/data_preration/data/available_embedding/0.5s_TTS_concat/test-clean
 - 61-70968-0041.npy
 - 121-121726-0010.npy
 - 237-126133-0000.npy
 - ...
```