import os
import asyncio
import json
import os
import queue
import threading
from queue import Queue
import time
import cv2
import numpy as np
import torch
import tqdm
import resampy
import edge_tts
from asyncio import Event
from io import BytesIO
from av import AudioFrame
from av.video.frame import VideoFrame
from transformers import Wav2Vec2Processor, AutoProcessor, AutoModelForCTC, HubertModel
from core.stream_track import AUDIO_PTIME, SAMPLE_RATE
import soundfile as sf
import torch.nn.functional as F
from torch.utils.data import DataLoader

from talkers.synctalk.config import set_opt
from talkers.synctalk.nerf_triplane.network import NeRFNetwork, AudioEncoder
from talkers.synctalk.nerf_triplane.provider import nerf_matrix_to_ngp, smooth_camera_path
from talkers.synctalk.nerf_triplane.utils import get_bg_coords, get_audio_features, get_rays, seed_everything, Trainer, \
    AudDataset


class SyncTalkDataset:
    def __init__(self, opt, device, downscale=1):
        super().__init__()

        self.opt = opt
        self.root_path = opt.root_path
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

        # only load one specified split
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
        print(f'[INFO] load {len(frames)} {type} frames.')

        bs = np.load(os.path.join(self.root_path, 'bs.npy'))
        if self.opt.bs_area == "upper":
            bs = np.hstack((bs[:, 0:5], bs[:, 8:10]))
        elif self.opt.bs_area == "single":
            bs = np.hstack(
                (bs[:, 0].reshape(-1, 1), bs[:, 2].reshape(-1, 1), bs[:, 3].reshape(-1, 1), bs[:, 8].reshape(-1, 1)))
        elif self.opt.bs_area == "eye":
            bs = bs[:, 8:10]

        self.torso_img = []
        self.images = []
        self.gt_images = []
        self.face_mask_imgs = []

        self.poses = []
        self.exps = []

        self.auds = []
        self.face_rect = []
        self.lhalf_rect = []
        self.upface_rect = []
        self.lowface_rect = []
        self.lips_rect = []
        self.eye_area = []
        self.eye_rect = []

        for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):

            f_path = os.path.join(self.root_path, 'gt_imgs', str(f['img_id']) + '.jpg')

            if not os.path.exists(f_path):
                print('[WARN]', f_path, 'NOT FOUND!')
                continue

            pose = np.array(f['transform_matrix'], dtype=np.float32)  # [4, 4]
            pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)
            self.poses.append(pose)

            if self.preload > 0:
                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype(np.float32) / 255  # [H, W, 3/4]

                self.images.append(image)
            else:
                self.images.append(f_path)

            if self.opt.portrait:
                gt_path = os.path.join(self.root_path, 'ori_imgs', str(f['img_id']) + '.jpg')
                # gt_path = os.path.join(self.root_path, 'torso_imgs', str(f['img_id']) + '_no_face.png')
                if not os.path.exists(f_path):
                    print('[WARN]', f_path, 'NOT FOUND!')
                    continue
                if self.preload > 0:
                    gt_image = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
                    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
                    gt_image = gt_image.astype(np.float32) / 255  # [H, W, 3/4]

                    self.gt_images.append(gt_image)
                else:
                    self.gt_images.append(gt_path)

                face_mask_path = os.path.join(self.root_path, 'parsing', str(f['img_id']) + '_face.png')
                if not os.path.exists(face_mask_path):
                    print('[WARN]', face_mask_path, 'NOT FOUND!')
                    continue
                if self.preload > 0:
                    face_mask_img = (255 - cv2.imread(face_mask_path)[:, :, 1]) / 255.0
                    self.face_mask_imgs.append(face_mask_img)
                else:
                    self.face_mask_imgs.append(face_mask_path)

            torso_img_path = os.path.join(self.root_path, 'torso_imgs', str(f['img_id']) + '.png')
            if self.preload > 0:
                torso_img = cv2.imread(torso_img_path, cv2.IMREAD_UNCHANGED)  # [H, W, 4]
                torso_img = cv2.cvtColor(torso_img, cv2.COLOR_BGRA2RGBA)
                torso_img = torso_img.astype(np.float32) / 255  # [H, W, 3/4]
                self.torso_img.append(torso_img)
            else:
                self.torso_img.append(torso_img_path)

            # load lms and extract face
            lms = np.loadtxt(os.path.join(self.root_path, 'ori_imgs', str(f['img_id']) + '.lms'))  # [68, 2]

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max())  # actually lower half area
            upface_xmin, upface_xmax = int(lms[:, 1].min()), int(lms[30, 1])
            lowface_xmin, lowface_xmax = int(lms[30, 1]), int(lms[:, 1].max())
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            self.face_rect.append([xmin, xmax, ymin, ymax])
            self.lhalf_rect.append([lh_xmin, lh_xmax, ymin, ymax])
            self.upface_rect.append([upface_xmin, upface_xmax, ymin, ymax])
            self.lowface_rect.append([lowface_xmin, lowface_xmax, ymin, ymax])

            if self.opt.exp_eye:
                area = bs[f['img_id']]
                self.eye_area.append(area)

                xmin, xmax = int(lms[36:48, 1].min()), int(lms[36:48, 1].max())
                ymin, ymax = int(lms[36:48, 0].min()), int(lms[36:48, 0].max())
                self.eye_rect.append([xmin, xmax, ymin, ymax])

            if self.opt.finetune_lips:
                lips = slice(48, 60)
                xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
                ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

                # padding to H == W
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2

                l = max(xmax - xmin, ymax - ymin) // 2
                xmin = max(0, cx - l)
                xmax = min(self.H, cx + l)
                ymin = max(0, cy - l)
                ymax = min(self.W, cy + l)

                self.lips_rect.append([xmin, xmax, ymin, ymax])

        # load pre-extracted background image (should be the same size as training image...)

        if self.opt.bg_img == 'white':  # special
            bg_img = np.ones((self.H, self.W, 3), dtype=np.float32)
        elif self.opt.bg_img == 'black':  # special
            bg_img = np.zeros((self.H, self.W, 3), dtype=np.float32)
        else:  # load from file
            # default bg
            if self.opt.bg_img == '':
                self.opt.bg_img = os.path.join(self.root_path, 'bc.jpg')
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

        if self.preload > 0:
            self.torso_img = torch.from_numpy(np.stack(self.torso_img, axis=0))  # [N, H, W, C]
            if self.opt.portrait:
                self.gt_images = torch.from_numpy(np.stack(self.gt_images, axis=0))  # [N, H, W, C]
                self.face_mask_imgs = torch.from_numpy(np.stack(self.face_mask_imgs, axis=0))  # [N, H, W, C]

        else:
            self.torso_img = np.array(self.torso_img)
            if self.opt.portrait:
                self.gt_images = np.array(self.gt_images)
                self.face_mask_imgs = np.array(self.face_mask_imgs)

        # live streaming, no pre-calculated auds
        self.auds = None

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
            if self.opt.bs_area == "upper":
                self.eye_area = torch.from_numpy(self.eye_area).view(-1, 7)  # [N, 7]
            elif self.opt.bs_area == "single":
                self.eye_area = torch.from_numpy(self.eye_area).view(-1, 4)  # [N, 7]
            else:
                self.eye_area = torch.from_numpy(self.eye_area).view(-1, 2)

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()

        if self.preload > 1:
            self.poses = self.poses.to(self.device)

            if self.auds is not None:
                self.auds = self.auds.to(self.device)

            self.bg_img = self.bg_img.to(torch.half).to(self.device)

            self.torso_img = self.torso_img.to(torch.half).to(self.device)  # todo
            # self.images = self.images.to(torch.half).to(self.device)
            if self.opt.portrait:
                self.gt_images = self.gt_images.to(torch.half).to(self.device)
                self.face_mask_imgs = self.face_mask_imgs.to(torch.half).to(self.device)

            if self.opt.exp_eye:
                self.eye_area = self.eye_area.to(self.device)

        # load intrinsics
        if 'focal_len' in transform:
            fl_x = fl_y = transform['focal_len']
        elif 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)

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
        results['up_rect'] = self.upface_rect[index[0]]
        results['low_rect'] = self.lowface_rect[index[0]]
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
        bg_torso_img = self.torso_img[index]
        if self.preload == 0:  # on the fly loading
            bg_torso_img = cv2.imread(bg_torso_img[0], cv2.IMREAD_UNCHANGED)  # [H, W, 4]
            bg_torso_img = cv2.cvtColor(bg_torso_img, cv2.COLOR_BGRA2RGBA)
            bg_torso_img = bg_torso_img.astype(np.float32) / 255  # [H, W, 3/4]
            bg_torso_img = torch.from_numpy(bg_torso_img).unsqueeze(0)
        bg_torso_img = bg_torso_img[..., :3] * bg_torso_img[..., 3:] + self.bg_img * (1 - bg_torso_img[..., 3:])
        bg_torso_img = bg_torso_img.view(B, -1, 3).to(self.device)

        if not self.opt.torso:
            bg_img = bg_torso_img  # 不推理
        else:
            bg_img = self.bg_img.view(1, -1, 3).repeat(B, 1, 1).to(self.device)


        results['bg_color'] = bg_img
        if self.opt.portrait:
            bg_gt_images = self.gt_images[index]
            if self.preload == 0:
                bg_gt_images = cv2.imread(bg_gt_images[0], cv2.IMREAD_UNCHANGED)
                bg_gt_images = cv2.cvtColor(bg_gt_images, cv2.COLOR_BGR2RGB)
                bg_gt_images = bg_gt_images.astype(np.float32) / 255
                bg_gt_images = torch.from_numpy(bg_gt_images).unsqueeze(0)
            bg_gt_images = bg_gt_images.to(self.device)
            results['bg_gt_images'] = bg_gt_images

            bg_face_mask = self.face_mask_imgs[index]
            if self.preload == 0:
                # bg_face_mask = np.all(cv2.imread(bg_face_mask[0]) == [255, 0, 0], axis=-1).astype(np.uint8)
                bg_face_mask = (255 - cv2.imread(bg_face_mask[0])[:, :, 1]) / 255.0
                bg_face_mask = torch.from_numpy(bg_face_mask).unsqueeze(0)
            bg_face_mask = bg_face_mask.to(self.device)
            results['bg_face_mask'] = bg_face_mask

        # images = self.images[index]  # [B, H, W, 3/4]
        # if self.preload == 0:
        #     images = cv2.imread(images[0], cv2.IMREAD_UNCHANGED)  # [H, W, 3]
        #     images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        #     images = images.astype(np.float32) / 255  # [H, W, 3]
        #     images = torch.from_numpy(images).unsqueeze(0)
        # images = images.to(self.device)
        # results['images'] = images
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
        elif 'hubert' in self.opt.asr_model:
            self.audio_dim = 1024
        # elif 'ave' in self.opt.asr_model:
        #     self.audio_dim = 512
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
        # if 'ave' in opt.asr_model:
        #     self.model = AudioEncoder().to(self.device).eval()
        #     ckpt = torch.load(opt.audio_visual_encoder)
        #     self.model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
        # else:
        if 'hubert' in self.opt.asr_model:
            self.processor = Wav2Vec2Processor.from_pretrained(opt.asr_model)
            self.model = HubertModel.from_pretrained(opt.asr_model).to(self.device)
        else:
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
        # if 'ave' in self.opt.asr_model:
        #     dataset = AudDataset(wav=frame)
        #     data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        #     outputs = []
        #     for mel in data_loader:
        #         mel = mel.to(self.device)
        #         with torch.no_grad():
        #             out = self.model(mel)
        #         outputs.append(out)
        #     outputs = torch.cat(outputs, dim=0).cpu()
        #     first_frame, last_frame = outputs[:1], outputs[-1:]
        #     aud_features = torch.cat([first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)], dim=0).numpy()
        #
        #     aud_features = torch.from_numpy(aud_features).unsqueeze(0)
        #     if len(aud_features.shape) == 3:
        #         aud_features = aud_features.float().permute(1, 0, 2)  # n, 1, 512
        #     get_audio_features(aud_features, self.opt.att, index[0]).to(self.device)


        # else:
        inputs = self.processor(frame, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            result = self.model(inputs.input_values.to(self.device))
            if 'hubert' in self.opt.asr_model:
                logits = result.last_hidden_state
            else:
                logits = result.logits # [1, N - 1, 32]

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
            # print('[INFO] get Empty audio frame')
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


class SyncTalkLink:
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
        data_loader = SyncTalkDataset(opt, device=device).dataloader()  # DataSet
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
        self.asr.warm_up()

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
            n_frame = int(audio_chunk_num / 2.) + 1
            # n_frame = int(audio_chunk_num * (0.02*(25)))
            # print('输入总帧数： ', n_frame)

            # calculate time block
            # 总帧数
            # n = len(filelist)
            #
            # 计算相差帧数
            # audio_time = int((n_frame / 25)) + 1
            audio_time = n_frame / 25.
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
                if self.opt.full_body:  # todo dataloader
                    full_body_img = cv2.imread(os.path.join(self.opt.full_body_imgs, str(data['index'][0])+'.jpg'))
                    full_body_img = cv2.cvtColor(full_body_img, cv2.COLOR_BGR2RGB)
                    full_body_img[self.opt.crop_y:self.opt.crop_y + image.shape[0], self.opt.crop_x:self.opt.crop_x + image.shape[1]] = image
                    video_frame = VideoFrame.from_ndarray(full_body_img, format="rgb24")
                else:
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

    def process_silence_template_video(self, output_path, num=300, start_idx=0):
        print('Generate silence template video...')
        template_video = []
        self.loader = iter(self.data_loader)
        min_len = min(len(self.data_loader), num)
        for i in tqdm.tqdm(range(min_len)):
            if i <= start_idx:
                continue
            start_idx += 1

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
            if self.opt.full_body:  # todo dataloader
                full_body_img = cv2.imread(os.path.join(self.opt.full_body_imgs, str(data['index'][0]) + '.jpg'))
                full_body_img = cv2.cvtColor(full_body_img, cv2.COLOR_BGR2RGB)
                full_body_img[self.opt.crop_y:self.opt.crop_y + image.shape[0],
                self.opt.crop_x:self.opt.crop_x + image.shape[1]] = image
                template_video.append(full_body_img)
            else:
                template_video.append(image)
        self.loader = iter(self.data_loader)
        try:
            np.save(output_path, np.asarray(template_video))
        except Exception as e:
            print('failed to save template video: ', e)


