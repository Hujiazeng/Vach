# Vach: Real-Time stream talking head
流式数字人，实现音视频同步对话。基本可以达到商用效果


[//]: # (## 🔥🔥🔥 Features)
## Features
- [x] **解决声音卡顿问题**
- [x] **监听麦克风输入**
- [x] **语言模型回复**
- [ ] **声音克隆**
- [x] **SyncTalk项目支持**
- [ ] **直播间业务**
- [ ] **展厅显示屏互动**

### Installation

Tested on Ubuntu 18.04, Pytorch 1.12.1 and CUDA 11.3.
```bash
git clonehttps://github.com/Hujiazeng/Vach.git
cd Vach
```
#### Install dependency

```bash
conda create -n Vach python==3.10
conda activate Vach
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html
pip install ./freqencoder
pip install ./shencoder
pip install ./gridencoder
pip install ./raymarching
```

#### 数字人模型文件
可以替换成自己训练的模型(https://github.com/Fictionarry/ER-NeRF)
```python
.
├── data
│   ├── obama(user-defined)
│       ├── transforms_train.json
│       ├── au.csv			
│       ├── ngp_kf.pth
│       ├── template.npy

```


### Quick Start

[//]: # (#### Prepare)


```python
python app.py
```

如果访问不了huggingface，在运行前
```
export HF_ENDPOINT=https://hf-mirror.com
```

用浏览器打开http://127.0.0.1:8010/webrtc.html, 建立连接后, 在文本框提交任何文字。 


如果项目对你有帮助，帮忙点个star。也欢迎感兴趣的朋友一起来完善该项目。

微信：hairong0907 加我进交流群


## Acknowledgement
This code is developed heavily relying on [aiortc](https://github.com/aiortc/aiortc), and also [ER-NeRF](https://github.com/Fictionarry/ER-NeRF) and  [SyncTalk](https://github.com/ZiqiaoPeng/SyncTalk).

Thanks for these great projects.


