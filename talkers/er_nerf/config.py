import os


def set_opt(opt):
    # Custom Settings
    data_path = os.path.join(opt.base_dir, "talkers", "er_nerf", "data", opt.model_name)
    opt.asr_model = 'cpierse/wav2vec2-large-xlsr-53-esperanto' # r'C:\Users\k\.cache\huggingface\hub\models--cpierse--wav2vec2-large-xlsr-53-esperanto\snapshots\5ae6c3174dddd1261dd2816eeb6f060655042026'
    opt.preload = 2
    opt.pose = os.path.join(data_path, "transforms_train.json")
    opt.au = os.path.join(data_path, "au.csv")
    opt.ckpt = os.path.join(data_path, "ngp_kf.pth")
    opt.template = os.path.join(data_path, "template.npy")
    opt.bg_img = "white"  #  os.path.join(data_path, "bc.jpg")  # "white"
    opt.torso_imgs = r''
    opt.W = 450  # 预处理crop时的原始大小 256
    opt.H = 450

    # full_body
    opt.full_body = False
    opt.full_body_imgs = r''
    opt.crop_x, opt.crop_y = 367, 276

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
    opt.torso = False
    if opt.torso_imgs == '':  # no img,use model output
        opt.torso = True
    return opt