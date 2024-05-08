import asyncio
import json
import os
import queue
import threading
from queue import Queue
import time
from asyncio import Event
from io import BytesIO

import cv2
import edge_tts
import resampy
import numpy as np
import torch
from av import AudioFrame
from av.video.frame import VideoFrame
import soundfile as sf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Processor, AutoProcessor, AutoModelForCTC
import pandas as pd
import torch.nn.functional as F

from core.stream_track import AUDIO_PTIME, SAMPLE_RATE
from talkers.er_nerf.nerf_triplane.network import NeRFNetwork
from talkers.er_nerf.nerf_triplane.provider import nerf_matrix_to_ngp, smooth_camera_path
from talkers.er_nerf.nerf_triplane.utils import get_bg_coords, get_audio_features, get_rays, seed_everything, Trainer


def set_opt(opt):
    # Custom Settings
    data_path = os.path.join(opt.base_dir, "talkers", opt.link_name, "data", opt.model_name)
    opt.asr_model = 'cpierse/wav2vec2-large-xlsr-53-esperanto' # r'C:\Users\k\.cache\huggingface\hub\models--cpierse--wav2vec2-large-xlsr-53-esperanto\snapshots\5ae6c3174dddd1261dd2816eeb6f060655042026'
    opt.preload = 2
    opt.pose = os.path.join(data_path, "transforms_train.json")
    opt.au = os.path.join(data_path, "au.csv")
    opt.ckpt = os.path.join(data_path, "ngp_kf.pth")
    opt.template = os.path.join(data_path, "template.npy")
    opt.bg_img = "white"
    opt.torso_imgs = ''
    opt.W = 450
    opt.H = 450

    # fixed
    opt.num_rays = 65536
    opt.fbg = False
    opt.dt_gamma = 1 / 256  # todo 加速效果非常明显  默认1/256
    opt.O = False
    opt.amb_aud_loss = 1
    opt.amb_dim = 2
    opt.amb_eye_loss = 1
    opt.asr = True
    opt.asr_save_feats = False
    opt.radius = 3.35
    opt.seed = 0
    opt.gui = False
    opt.fovy = 21.24
    opt.lambda_amb = 0.0001
    opt.max_ray_batch = 4096
    opt.max_spp = 1
    opt.max_steps = 16
    opt.part = False
    opt.part2 = False
    opt.test = True
    opt.update_extra_interval = 16
    opt.upsample_steps = 0
    opt.warmup_step = 10000
    opt.fp16 = True
    opt.color_space = 'srgb'  # 尝试更换
    opt.fps = 50  # 25??
    opt.finetune_lips = False
    opt.asr_wav = ''
    opt.smooth_path = True
    opt.cuda_ray = True
    opt.asr_play = False
    opt.scale = 4
    opt.torso_shrink = 0.8
    opt.l = 10
    opt.m = 8
    opt.r = 10
    opt.fix_eye = -1
    opt.aud = ''
    opt.offset = [0, 0, 0]
    opt.data_range = [0, -1]
    opt.init_lips = False
    opt.patch_size = 1
    opt.exp_eye = True
    opt.smooth_eye = True
    opt.asr = True
    opt.bound = 1
    opt.min_near = 0.05
    opt.density_thresh = 10
    opt.density_thresh_torso = 0.01
    opt.test_train = False
    opt.smooth_lips = True
    opt.smooth_path_window = 7
    opt.ind_dim = 4
    opt.ind_num = 10000
    opt.ind_dim_torso = 8
    opt.train_camera = False
    opt.emb = False
    opt.att = 2
    if opt.torso_imgs == '':  # no img,use model output
        opt.torso = True
    return opt


class NerfTestDataset:
    def __init__(self, opt, device, downscale=1):
        self.opt = opt
        self.device = device
        self.downscale = downscale
        self.preload = opt.preload  # 0 = disk, 1 = cpu, 2 = gpu
        self.scale = opt.scale  # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset  # camera offset
        self.bound = opt.bound  # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16

        self.start_index = opt.data_range[0]
        self.end_index = opt.data_range[1]
        self.training = False
        self.num_rays = -1

        with open(opt.pose, 'r') as f:
            transform = json.load(f)

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            self.H = int(transform['cy']) * 2 // downscale
            self.W = int(transform['cx']) * 2 // downscale

        # read images
        frames = transform["frames"]
        # use a slice of the dataset
        if self.end_index == -1:  # abuse...
            self.end_index = len(frames)

        frames = frames[self.start_index:self.end_index]
        print(f'[INFO] load {len(frames)} frames.')

        # load action units
        au_blink_info = pd.read_csv(self.opt.au)
        au_blink = au_blink_info[' AU45_r'].values

        self.poses = []
        self.auds = None
        self.eye_area = []
        self.torso_img = []
        for f in tqdm(frames, desc=f'Loading {type} data'):
            pose = np.array(f['transform_matrix'], dtype=np.float32)  # [4, 4]
            pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
            self.poses.append(pose)

            if self.opt.torso_imgs:
                torso_img_path = os.path.join(self.opt.torso_imgs, str(f['img_id']) + '.png')
                if self.preload > 0:
                    torso_img = cv2.imread(torso_img_path, cv2.IMREAD_UNCHANGED)  # [H, W, 4]
                    torso_img = cv2.cvtColor(torso_img, cv2.COLOR_BGRA2RGBA)
                    torso_img = torso_img.astype(np.float32) / 255  # [H, W, 3/4]
                    self.torso_img.append(torso_img)
                else:
                    self.torso_img.append(torso_img_path)

            if self.opt.exp_eye:
                # action units blink AU45
                area = au_blink[f['img_id']]
                area = np.clip(area, 0, 2) / 2
                # area = area + np.random.rand() / 10
                self.eye_area.append(area)

        if self.opt.torso_imgs:
            if self.preload > 0:
                self.torso_img = torch.from_numpy(np.stack(self.torso_img, axis=0))  # [N, H, W, C]
            else:
                self.torso_img = np.array(self.torso_img)
            if self.preload > 1:  # gpu
                self.torso_img = self.torso_img.to(torch.half).to(self.device)

        # load pre-extracted background image (should be the same size as training image...)

        if self.opt.bg_img == 'white':  # special
            bg_img = np.ones((self.H, self.W, 3), dtype=np.float32)
        elif self.opt.bg_img == 'black':  # special
            bg_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        else:  # load from file
            # default bg
            bg_img = cv2.imread(self.opt.bg_img, cv2.IMREAD_UNCHANGED)  # [H, W, 3]
            if bg_img.shape[0] != self.H or bg_img.shape[1] != self.W:
                bg_img = cv2.resize(bg_img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = bg_img.astype(np.float32) / 255  # [H, W, 3/4]

        self.bg_img = bg_img

        self.poses = np.stack(self.poses, axis=0)

        # smooth camera path...
        if self.opt.smooth_path:
            self.poses = smooth_camera_path(self.poses, self.opt.smooth_path_window)

        self.poses = torch.from_numpy(self.poses)  # [N, 4, 4]

        # if self.preload > 0:
        #     self.torso_img = torch.from_numpy(np.stack(self.torso_img, axis=0))  # [N, H, W, C]
        # else:
        #     self.torso_img = np.array(self.torso_img)

        self.bg_img = torch.from_numpy(self.bg_img)

        if self.opt.exp_eye:
            self.eye_area = np.array(self.eye_area, dtype=np.float32)  # [N]
            print(f'[INFO] eye_area: {self.eye_area.min()} - {self.eye_area.max()}')

            if self.opt.smooth_eye:
                # naive 5 window average
                ori_eye = self.eye_area.copy()
                for i in range(ori_eye.shape[0]):
                    start = max(0, i - 1)
                    end = min(ori_eye.shape[0], i + 2)
                    self.eye_area[i] = ori_eye[start:end].mean()
            self.eye_area = torch.from_numpy(self.eye_area).view(-1, 1)  # [N, 1]

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()

        if self.preload > 1 or self.opt.torso_imgs == '':  # todo 为什么不能一起加速
            self.bg_img = self.bg_img.to(torch.half).to(self.device)
            # self.torso_img = self.torso_img.to(torch.half).to(self.device)

        self.poses = self.poses.to(self.device)
        if self.opt.exp_eye:
            self.eye_area = self.eye_area.to(self.device)

        # load intrinsics
        fl_x = fl_y = transform['focal_len']
        cx = (transform['cx'] / downscale)
        cy = (transform['cy'] / downscale)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        # directly build the coordinate meshgrid in [-1, 1]^2
        self.bg_coords = get_bg_coords(self.H, self.W, self.device)  # [1, H*W, 2] in [-1, 1]

    def mirror_index(self, index):
        size = self.poses.shape[0]
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    def collate(self, index):

        B = len(index)  # a list of length 1
        # assert B == 1

        results = {}

        # audio use the original index
        if self.auds is not None:
            auds = get_audio_features(self.auds, self.opt.att, index[0]).to(self.device)
            results['auds'] = auds

        # head pose and bg image may mirror (replay --> <-- --> <--).
        index[0] = self.mirror_index(index[0])

        poses = self.poses[index].to(self.device)  # [B, 4, 4]
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, self.opt.patch_size)

        results['index'] = index  # for ind. code
        results['H'] = self.H
        results['W'] = self.W
        results['rays_o'] = rays['rays_o']
        results['rays_d'] = rays['rays_d']

        if self.opt.exp_eye:
            results['eye'] = self.eye_area[index].to(self.device)  # [1]
        else:
            results['eye'] = None

        # load bg
        if self.opt.torso_imgs != '':
            bg_torso_img = self.torso_img[index]
            if self.preload == 0:  # on the fly loading
                bg_torso_img = cv2.imread(bg_torso_img[0], cv2.IMREAD_UNCHANGED)  # [H, W, 4]
                bg_torso_img = cv2.cvtColor(bg_torso_img, cv2.COLOR_BGRA2RGBA)
                bg_torso_img = bg_torso_img.astype(np.float32) / 255  # [H, W, 3/4]
                bg_torso_img = torch.from_numpy(bg_torso_img).unsqueeze(0)
            bg_torso_img = bg_torso_img[..., :3] * bg_torso_img[..., 3:] + self.bg_img * (1 - bg_torso_img[..., 3:])
            bg_torso_img = bg_torso_img.view(B, -1, 3).to(self.device)

            if not self.opt.torso:
                bg_img = bg_torso_img
            else:
                bg_img = self.bg_img.view(1, -1, 3).repeat(B, 1, 1).to(self.device)
        else:
            bg_img = self.bg_img.view(1, -1, 3).repeat(B, 1, 1).to(self.device)

        results['bg_color'] = bg_img

        bg_coords = self.bg_coords  # [1, N, 2]
        results['bg_coords'] = bg_coords
        results['poses'] = poses  # [B, 4, 4]
        return results

    def dataloader(self):
        # test with novel auds, then use its length
        if self.auds is not None:
            size = self.auds.shape[0]
        # live stream test, use 2 * len(poses), so it naturally mirrors.
        else:
            size = 2 * self.poses.shape[0]

        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training,
                            num_workers=0)
        loader._data = self  # an ugly fix... we need poses in trainer.

        # do evaluate if has gt images and use self-driven setting
        loader.has_gt = (self.opt.aud == '')

        return loader


class ASR:
    def __init__(self, opt):
        self.opt = opt

        self.play = opt.asr_play

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fps = opt.fps  # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps  # 320 samples per chunk (20ms * 16000 / 1000)
        self.mode = 'live' if opt.asr_wav == '' else 'file'

        if 'esperanto' in self.opt.asr_model:
            self.audio_dim = 44
        elif 'deepspeech' in self.opt.asr_model:
            self.audio_dim = 29
        else:
            self.audio_dim = 32

        # prepare context cache
        # each segment is (stride_left + ctx + stride_right) * 20ms, latency should be (ctx + stride_right) * 20ms
        self.context_size = opt.m
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r
        self.text = '[START]\n'
        self.terminated = False
        self.frames = []
        self.inwarm = False

        # pad left frames
        if self.stride_left_size > 0:
            self.frames.extend([np.zeros(self.chunk, dtype=np.float32)] * self.stride_left_size)

        self.exit_event = Event()
        # create input stream
        self.queue = Queue()
        self.input_stream = BytesIO()
        self.output_queue = Queue()

        # current location of audio
        self.idx = 0

        # create wav2vec model
        print(f'[INFO] loading ASR model {self.opt.asr_model}...')
        self.processor = AutoProcessor.from_pretrained(opt.asr_model)
        self.model = AutoModelForCTC.from_pretrained(opt.asr_model).to(self.device)

        # prepare to save logits
        # if self.opt.asr_save_feats:
        #     self.all_feats = []

        # the extracted features
        # use a loop queue to efficiently record endless features: [f--t---][-------][-------]
        self.feat_buffer_size = 4
        self.feat_buffer_idx = 0
        self.feat_queue = torch.zeros(self.feat_buffer_size * self.context_size, self.audio_dim,
                                      dtype=torch.float32,
                                      device=self.device)

        # TODO: hard coded 16 and 8 window size...
        self.front = self.feat_buffer_size * self.context_size - 8  # fake padding
        self.tail = 8
        # attention window...
        self.att_feats = [torch.zeros(self.audio_dim, 16, dtype=torch.float32,
                                      device=self.device)] * 4  # 4 zero padding...

        # warm up steps needed: mid + right + window_size + attention_size
        self.warm_up_steps = self.context_size + self.stride_left_size + self.stride_right_size + 8 + 2 * 3

        self.listening = False
        self.playing = False

    def get_next_feat(self):  # nerf get next audio feat
        # return a [1/8, 16] window, for the next input to nerf side.
        if self.opt.att > 0:
            while len(self.att_feats) < 8:
                # [------f+++t-----]
                if self.front < self.tail:
                    feat = self.feat_queue[self.front:self.tail]
                # [++t-----------f+]
                else:
                    feat = torch.cat([self.feat_queue[self.front:], self.feat_queue[:self.tail]], dim=0)

                self.front = (self.front + 2) % self.feat_queue.shape[0]
                self.tail = (self.tail + 2) % self.feat_queue.shape[0]

                # print(self.front, self.tail, feat.shape)

                self.att_feats.append(feat.permute(1, 0))

            att_feat = torch.stack(self.att_feats, dim=0)  # [8, 44, 16]

            # discard old
            self.att_feats = self.att_feats[1:]
        else:
            # [------f+++t-----]
            if self.front < self.tail:
                feat = self.feat_queue[self.front:self.tail]
            # [++t-----------f+]
            else:
                feat = torch.cat([self.feat_queue[self.front:], self.feat_queue[:self.tail]], dim=0)

            self.front = (self.front + 2) % self.feat_queue.shape[0]
            self.tail = (self.tail + 2) % self.feat_queue.shape[0]

            att_feat = feat.permute(1, 0).unsqueeze(0)

        return att_feat

    def get_audio_out(self):  # get origin audio pcm to nerf
        return self.output_queue.get()

    def frame_to_text(self, frame):
        # frame: [N * 320], N = (context_size + 2 * stride_size)

        inputs = self.processor(frame, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)

        with torch.no_grad():
            result = self.model(inputs.input_values.to(self.device))
            if 'hubert' in self.opt.asr_model:
                logits = result.last_hidden_state # [B=1, T=pts//320, hid=1024]
            else:
                logits = result.logits  # [1, N - 1, 32]

        # cut off stride
        left = max(0, self.stride_left_size)
        right = min(logits.shape[1],
                    logits.shape[
                        1] - self.stride_right_size + 1)  # +1 to make sure output is the same length as input.

        # do not cut right if terminated.
        if self.terminated:
            right = logits.shape[1]

        logits = logits[:, left:right]

        return logits[0]  # [N,]

    def get_audio_frame(self):
        if self.inwarm:  # warm up  # todo
            return np.zeros(self.chunk, dtype=np.float32)
        try:
            frame = self.queue.get(block=False)
            # print(f'[INFO] get audio frame {frame.shape}')
        except queue.Empty:
            frame = np.zeros(self.chunk, dtype=np.float32)  # 无音频填充空数据
            print('[INFO] get Empty audio frame')
        self.idx = self.idx + self.chunk
        return frame

    def run_step(self):

        if self.terminated:
            return

        # get a frame of audio
        frame = self.get_audio_frame()
        self.frames.append(frame)
        # put to output
        self.output_queue.put(frame)
        # context not enough, do not run network.
        if len(self.frames) < self.stride_left_size + self.context_size + self.stride_right_size:
            return

        inputs = np.concatenate(self.frames)  # [N * chunk]
        # discard the old part to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]
        logits = self.frame_to_text(inputs)
        feats = logits  # better lips-sync than labels

        # record the feats efficiently.. (no concat, constant memory)
        start = self.feat_buffer_idx * self.context_size
        end = start + feats.shape[0]
        self.feat_queue[start:end] = feats
        self.feat_buffer_idx = (self.feat_buffer_idx + 1) % self.feat_buffer_size

    def warm_up(self):
        # self.listen()
        self.inwarm = True
        print(f'[INFO] warm up ASR live model, expected latency = {self.warm_up_steps / self.fps:.6f}s')
        t = time.time()
        # for _ in range(self.stride_left_size):
        #     self.frames.append(np.zeros(self.chunk, dtype=np.float32))
        for _ in range(self.warm_up_steps):
            self.run_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.time() - t
        print(f'[INFO] warm-up done, actual latency = {t:.6f}s')
        self.inwarm = False


class ErNerfLink:
    def __init__(self, opt, *args):
        super().__init__(*args)
        self.audio_track = None
        self.video_track = None
        self.input_stream = BytesIO()
        self.opt = set_opt(opt)
        self.user_audio_list = []  # audio length
        seed_everything(opt.seed)
        self.W = self.opt.W
        self.H = self.opt.H
        print(opt)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NeRFNetwork(opt)
        criterion = torch.nn.MSELoss(reduction='none')
        metrics = []  # use no metric in GUI for faster initialization...
        self.trainer = Trainer('ngp', opt, model, device=device, workspace=None, criterion=criterion, fp16=opt.fp16,
                               metrics=metrics, use_checkpoint=opt.ckpt)
        data_loader = NerfTestDataset(opt, device=device).dataloader()  # DataSet
        model.aud_features = data_loader._data.auds
        model.eye_areas = data_loader._data.eye_area
        self.data_loader = data_loader
        # use dataloader's bg
        bg_img = data_loader._data.bg_img  # .view(1, -1, 3)
        if self.H != bg_img.shape[0] or self.W != bg_img.shape[1]:
            bg_img = F.interpolate(bg_img.permute(2, 0, 1).unsqueeze(0).contiguous(), (self.H, self.W),
                                   mode='bilinear').squeeze(0).permute(1, 2, 0).contiguous()
        self.bg_color = bg_img.view(1, -1, 3)
        # audio features (from dataloader, only used in non-playing mode)
        self.audio_features = data_loader._data.auds  # [N, 29, 16]
        self.audio_idx = 0
        # control eye
        self.eye_area = None if not self.opt.exp_eye else data_loader._data.eye_area.mean().item()
        # playing seq from dataloader, or pause.
        self.playing = True
        self.loader = iter(data_loader)
        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # camera moved, should reset accumulation
        self.spp = 1  # sample per pixel
        self.mode = 'image'  # choose from ['image', 'depth']
        self.dynamic_resolution = False  # assert False!
        self.downscale = 1
        self.train_steps = 16
        self.ind_index = 0
        self.ind_num = self.trainer.model.individual_codes.shape[0]
        # build asr
        self.asr = ASR(opt)

        # render thread
        self.loop = None
        self.render_blocks = []
        self.__render_thread, self.__render_thread_quit = None, None
        self.datasets = self.load_dataset()
        self.start_render_thread()

    def start_render_thread(self):
        if self.__render_thread is None:
            self.__render_thread_quit = threading.Event()
            self.__render_thread = threading.Thread(
                name="render",
                target=self.listen_and_render,
                args=(
                    asyncio.get_event_loop(),
                    self.__render_thread_quit,
                ),
            )
            self.__render_thread.start()

    def load_dataset(self):
        start_time = time.time()
        datasets = []
        min_len = min(len(self.data_loader), 300)
        for i in range(min_len):
            try:
                data = next(self.loader)
                datasets.append(data)
            except StopIteration:
                return datasets
        print('load dataset use time', time.time() - start_time)
        return datasets

    def get_video_stream(self):
        video_streams = []
        template_video = np.load(self.opt.template)
        for frame in template_video:
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_streams.append(video_frame)

        return video_streams

    def push_audio(self, byte_stream):
        if self.opt.tts == "edgetts":
            self.asr.input_stream.write(byte_stream)
            if len(byte_stream) <= 0:
                self.asr.input_stream.seek(0)
                stream = self.get_adapter_stream(self.asr.input_stream)  # 统一转换格式
                streamlen = stream.shape[0]
                idx = 0
                num = 0
                while streamlen >= self.asr.chunk:
                    self.asr.queue.put(stream[idx:idx + self.asr.chunk])
                    streamlen -= self.asr.chunk
                    idx += self.asr.chunk
                    num += 1

                self.user_audio_list.append(num)  # test
                # if streamlen>0:  #skip last frame(not 20ms)
                #    self.queue.put(stream[idx:])
                self.asr.input_stream.seek(0)  # 移动到开始位置
                self.asr.input_stream.truncate()  # 清空全部数据

    def listen_and_calculate_block(self, quit_event, loop=None):
        if self.loop is None:
            self.loop = loop

        self.asr.warm_up()

        while not quit_event.is_set():
            # t = time.time()
            if not self.user_audio_list:
                time.sleep(0.3)
                continue

            audio_chunk_num = self.user_audio_list.pop(0)  # 320 0.02s
            # n_frame = int(audio_chunk_num / 2) + 1
            n_frame = int(audio_chunk_num * (0.02*25))   # test
            # print('输入总帧数： ', n_frame)

            # calculate time block
            # 总帧数
            # n = len(filelist)
            #
            # 计算相差帧数
            # audio_time = int((n_frame / 25)) + 1
            audio_time = int((n_frame / 25))
            distance_frame = int((25 - self.opt.real_fps) * audio_time) + 25  # 默认+1s
            # 计算下一帧出现位置  _streams.
            # 上一段未播放完毕
            if self.video_track.blocks:  # [block1, clear, block2, clear block3]
                _, _, _, end_frame_idx, _ = self.video_track.blocks[-1]
                # 计算还剩多少帧未播放的test
                no_play_frame_nums = self.video_track.no_play_frame_nums
                if no_play_frame_nums > distance_frame:
                    distance_frame = 25  # 默认间隔1s

                # 以当前第0圈计算 上一段跑完重置圈数
                new_insert_frame_laps = (end_frame_idx + distance_frame) // self.video_track.stream_len
                new_insert_frame_idx = (end_frame_idx + distance_frame) % self.video_track.stream_len
                new_end_frame_laps = (end_frame_idx + distance_frame + n_frame) // self.video_track.stream_len
                new_end_frame_idx = (end_frame_idx + distance_frame + n_frame) % self.video_track.stream_len
            else:
                # 播放默认视频中
                new_insert_frame_laps = (self.video_track.train_data_idx + distance_frame) // self.video_track.stream_len
                new_insert_frame_idx = (self.video_track.train_data_idx + distance_frame) % self.video_track.stream_len
                new_end_frame_laps = (self.video_track.train_data_idx + distance_frame + n_frame) // self.video_track.stream_len
                new_end_frame_idx = (self.video_track.train_data_idx + distance_frame + n_frame) % self.video_track.stream_len

            print("add new block:  ", new_insert_frame_laps, new_insert_frame_idx, new_end_frame_laps, new_end_frame_idx,
                  n_frame)
            self.video_track.blocks.append([
                new_insert_frame_laps, new_insert_frame_idx, new_end_frame_laps, new_end_frame_idx, n_frame
            ])
            self.video_track.no_play_frame_nums += n_frame  # 总帧数加

            self.render_blocks.append([new_insert_frame_idx, n_frame])

    def listen_and_render(self, loop, quit_event2):

        while not quit_event2.is_set():
            if not self.render_blocks:
                time.sleep(0.3)
                continue

            count = 0
            insert_frame_idx, n_frame = self.render_blocks.pop(0)
            t = time.time()
            for i in range(insert_frame_idx, insert_frame_idx + n_frame):
                idx = i % self.video_track.stream_len
                for _ in range(2):  # run 2 ASR steps (audio is at 50FPS, video is at 25FPS)
                    self.asr.run_step()  # 音频特征提取放入feat_queue

                data = self.datasets[idx]
                # data = next(self.loader)

                # video submit
                data['auds'] = self.asr.get_next_feat()
                outputs = self.trainer.test_gui_with_data(data, self.W, self.H)
                image = (outputs['image'] * 255).astype(np.uint8)
                video_frame = VideoFrame.from_ndarray(image, format="rgb24")
                asyncio.run_coroutine_threadsafe(self.video_track._queue.put(video_frame), self.loop)
                # audio submit
                for _ in range(2):
                    frame = self.asr.get_audio_out()
                    frame = (frame * 32767).astype(np.int16)  # 16bit
                    audio_frame = AudioFrame(format='s16', layout='mono',
                                             samples=int(AUDIO_PTIME * SAMPLE_RATE))  # 16000/fps
                    audio_frame.planes[0].update(frame.tobytes())
                    audio_frame.sample_rate = 16000
                    asyncio.run_coroutine_threadsafe(self.audio_track._queue.put(audio_frame), self.loop)

                count += 1
            print(f"------block fps------ : {count / (time.time()-t) :.4f}")
            # print("Block Finished")

    def get_adapter_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream)
        # print(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            # print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]

        if sample_rate != self.asr.sample_rate and stream.shape[0] > 0:
            # change audio sample
            # print(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.asr.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.asr.sample_rate)
        return stream

    async def say(self, text, voicename="zh-CN-YunxiaNeural", tts_type="edgetts"):
        if tts_type == "edgetts":
            communicate = edge_tts.Communicate(text, voicename)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    self.push_audio(chunk["data"])

    def process_silence_template_video(self, output_path, num=300):
        print('Generate silence template video...')
        template_video = []
        self.loader = iter(self.data_loader)
        min_len = min(len(self.data_loader), num)
        for _ in tqdm(range(min_len)):
            try:
                data = next(self.loader)
            except StopIteration:
                self.loader = iter(self.data_loader)
                data = next(self.loader)

            for _ in range(2):  # run 2 ASR steps (audio is at 50FPS, video is at 25FPS)
                self.asr.run_step()  # 音频特征提取放入feat_queue
            data['auds'] = self.asr.get_next_feat()
            outputs = self.trainer.test_gui_with_data(data, self.W, self.H)
            image = (outputs['image'] * 255).astype(np.uint8)
            template_video.append(image)
        self.loader = iter(self.data_loader)
        try:
            np.save(output_path, np.asarray(template_video))
        except Exception as e:
            print('failed to save template video: ', e)


