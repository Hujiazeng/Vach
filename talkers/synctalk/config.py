import os


def set_opt(opt):
    # Custom Settings
    data_path = os.path.join(opt.base_dir, "talkers", 'synctalk', "data", opt.model_name)
    opt.root_path = data_path
    opt.ckpt = os.path.join(data_path, "ngp_kf.pth")
    opt.pose = os.path.join(data_path, "transforms_train.json")
    opt.template = os.path.join(data_path, "template.npy")
    # opt.audio_visual_encoder = os.path.join(opt.base_dir, "talkers", 'synctalk', "nerf_triplane", "checkpoints", "audio_visual_encoder.pth")
    opt.bg_img = os.path.join(data_path, "bc.jpg")  # os.path.join(data_path, "bc.jpg")  # "white"
    opt.preload = 0
    opt.W = 450  # 预处理crop时的原始大小
    opt.H = 450

    opt.portrait = True
    opt.asr_model = r'facebook/hubert-large-ls960-ft'  # 'cpierse/wav2vec2-large-xlsr-53-esperanto  deepspeech facebook/hubert-large-ls960-ft'
    opt.data_range = [0, 1000]

    # full body
    opt.full_body=False
    opt.full_body_imgs = os.path.join(data_path, "fullbody_imgs")
    opt.crop_x, opt.crop_y = 0, 0

    opt.torso = False
    # if opt.torso_imgs == '':  # no img,use model output
    #     opt.torso = True


    opt.dt_gamma = 1 / 256
    opt.O = False
    opt.test = True
    opt.test_train = False
    opt.au45 = False
    opt.seed = 0
    opt.num_rays = 4096 * 16
    opt.cuda_ray = True
    opt.upsample_steps = 0
    opt.update_extra_interval = 16
    opt.max_ray_batch = 4096
    opt.warmup_step = 10000
    opt.amb_aud_loss = 1
    opt.amb_eye_loss = 1
    opt.unc_loss = 1
    opt.lambda_amb = 0.0001
    opt.pyramid_loss = 0
    opt.fp16 = True
    opt.fbg = False
    opt.exp_eye = True
    opt.fix_eye = -1
    opt.smooth_eye = True
    opt.bs_area = "upper"
    opt.torso_shrink = 0.8
    opt.color_space = 'srgb'
    opt.bound = 1
    opt.scale = 4
    opt.offset = [0, 0, 0]
    opt.min_near = 0.05
    opt.density_thresh = 10
    opt.density_thresh_torso = 0.01
    opt.patch_size = 1
    opt.init_lips = False
    opt.finetune_lips = False
    opt.smooth_lips = True
    opt.gui = False
    opt.radius = 3.35
    opt.fovy = 21.24
    opt.max_spp = 1
    opt.att = 2
    opt.aud = ''
    opt.emb = False
    opt.ind_dim = 4
    opt.ind_num = 20000  # todo ernerf default=10000
    opt.ind_dim_torso = 8
    opt.amb_dim = 2
    opt.part = False
    opt.part2 = False
    opt.train_camera = False
    opt.smooth_path = True
    opt.smooth_path_window = 7
    opt.asr = True
    opt.asr_wav = ''
    opt.asr_play = False
    opt.asr_save_feats = False
    opt.fps = 50
    opt.l = 10
    opt.m = 50
    opt.r = 10


    return opt

