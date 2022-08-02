from os.path import join
import torch
import torch.nn as nn

from model.vivit import ViViT
from model.vivit_rgb import ViViT_RGB


import losses


def get_model_opts(cfg, N=10000):
    '''
    Return:
        vit_models = vit_model
        opts = [opt_vit]
        schs = [sch_]
    '''
    gen_name = cfg.model
    resume = cfg.resume
    optimizer_name = cfg.optimizer
    scheduler_name = cfg.scheduler
    in_channels = cfg.nframe * 3
    out_channels = 3
    input_size = cfg.input_size
    patch_size = cfg.patch_size
    frame_nums = cfg.nframe
    
    info = {}

    # ViViT vit_models
    if gen_name == 'vivit':
        vit_model = ViViT(input_size, patch_size, frame_nums).cuda()
    elif gen_name == 'vivit_rgb':
        vit_model = ViViT_RGB(input_size, patch_size, frame_nums).cuda()
    else:
        raise NotImplementedError    
    

    # optimizer and scheduler
    if optimizer_name == 'adamw':
        opt_vit = torch.optim.AdamW(
                    vit_model.parameters(),
                    lr=cfg.g_lr,
                    weight_decay=cfg.l2,
                    )
    elif optimizer_name == 'adam':
        opt_vit = torch.optim.Adam(
                    vit_model.parameters(),
                    lr=cfg.g_lr,
                    weight_decay=cfg.l2,
                    )
    else:
        raise NotImplementedError

    if scheduler_name == 'no':
        sch_ = None
    elif scheduler_name == 'cosine':
        sch_ = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=opt_vit,
                    T_max=int(cfg.epoch*N/cfg.batch_size),
                    eta_min=0,
                    )
    elif scheduler_name == 'cosinewr':
        sch_ = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer=opt_vit,
                    T_0=cfg.warm_cycle,
                    T_mult=2,
                    eta_min=0,
                    )

    else:
        raise NotImplementedError
    
    if resume is not None:
        ckpt = torch.load(cfg.resume, map_location=cfg.device)
        # ckpt = torch.load(cfg.resume)
        vit_model.load_state_dict(ckpt['net_vit'])
        opt_vit.load_state_dict(ckpt['opt_vit'])
        if sch_ is not None:
            if 'sch_' not in list(ckpt.keys()):
                raise Exception('Config and checkpoint do not match')
            sch_.load_state_dict(ckpt['sch_'])
            info['last_lr'] = ckpt['sch_']['_last_lr'][0]
            info['step'] = ckpt['sch_']['_step_count']
        else:
            if 'sch_' in list(ckpt.keys()):
                raise Exception('Config and checkpoint do not match')
        print(f'Pre-trained vit_models and opts have been loaded.')

    else:
        if cfg.init == 'original':
            weights_init = weights_init_original
        elif cfg.init == 'xavier':
            weights_init = weights_init_xavier
        else:
            weights_init = weights_init_normal
        vit_model.apply(weights_init)
        print('vit_model and discriminator are going to be trained from scratch.')
    
    return vit_model, opt_vit, sch_, info


def get_losses(cfg):
    '''
    Return:
        losses = [adversarial_loss,
                    discriminate_loss,
                    gradient_loss,
                    intensity_loss,
                    flow_loss]
    '''
    arr_loss = losses.arr_Loss().cuda()
    irr_loss = losses.irr_Loss().cuda()
    rot_loss = losses.rot_Loss().cuda()


    total_loss = [arr_loss,
              irr_loss,
              rot_loss]

    _ = [l.to(cfg.device) for l in total_loss]

    return total_loss


def weights_init_original(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def save_all(vit_model, opts, schs, save_path):
    opt_vit = opts
    sch_ = schs
    
    vit_model_dict = {
            'net_vit': vit_model.state_dict(),
            'opt_vit': opt_vit.state_dict(),
            'sch_': sch_.state_dict(),
            }
    torch.save(vit_model_dict, save_path)
    return None


def get_checkpoint_vit_model(cfg):
    save_path = join(cfg.save_prefix, cfg.checkpoint)
    gen_name = cfg.vit_model
    in_channels = cfg.nframe * 3    
    yolo_device = f'{cfg.device}:{cfg.gpu_num}'
    yolo_vit_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True, device=yolo_device)
    
    # ViViT vit_models
    if gen_name == 'vivit':
        vit_model = ViViT(input_dim=in_channels)
    elif gen_name == 'vivit_rgb':
        vit_model = ViViT_RGB(input_dim=in_channels)
    else:
        raise NotImplementedError  
    
    vit_model.eval()
    vit_model.to(cfg.device)
    vit_model.load_state_dict(torch.load(save_path,
                                map_location=cfg.device)['net_vit'])

    return yolo_vit_model, vit_model