##  语音服务介绍

该服务以modelscope funasr语音识别为基础

## Install
conda create -n funasr
conda activate funasr
pip install torch
pip install modelscope==1.5.2
pip install testresources==2.0.1
pip install websockets
pip install torchaudio
pip install FunASR==1.0.24

## Start server
#### 启动ASR服务
python -u ASR_server.py --host "0.0.0.0" --port 10197 --ngpu 0


