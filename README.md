# Vach: Real-Time stream talking head
流式数字人，实现音视频同步对话,基本可以达到商用效果

![test](./web/show.gif)
来自群友@不蠢不蠢 部署成功的视频展示

[//]: # (## 🔥🔥🔥 Features)
## Features
- [x] **文本交互**
- [x] **语音交互**
- [x] **SyncTalk项目支持**
- [ ] **声音克隆**
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
# Note the following modules. If installation is unsuccessful, you can navigate to the path and use pip install . or python setup.py install to compile and install.
# NeRF/freqencoder
# NeRF/gridencoder
# NeRF/raymarching
# NeRF/shencoder
```

#### 数字人模型文件
我们提供[预训练模型](https://github.com/Hujiazeng/Vach/releases/tag/ckpt)下载并测试

可以替换成自己训练的模型(https://github.com/Fictionarry/ER-NeRF)
```python
.
├── data
│   ├── obama(user-defined)
│       ├── transforms_train.json
│       ├── au.csv			
│       ├── ngp_kf.pth
│       ├── template.npy(首次运行自动生成)
│       ├── torso_imgs(仅全身推理时使用)
│       ├── fullbody_imgs(仅全身推理时使用)

```


### Quick Start
```python
python app.py
```
#### 开启麦克风监听功能
```python
python app.py --mike
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


