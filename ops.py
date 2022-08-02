# Trainer and inference funcs
import logging

import torch
import numpy as np

import data_utils
import crop_utils

import random

logger = logging.getLogger('advision')


def detect_nob(last_frame, yolo_model, cfg, confidence=.4, coi=[0]):
    '''
    No bactch allowed.
    Return:
        areas_filters: list of np.arrays of [xmin, ymin, xmax, ymax, conf, class]
    '''
    results = yolo_model(last_frame)
    areas = results.xyxy[0]
    if cfg.device != 'cpu':
        areas = areas.cpu().numpy()
    else:
        areas = areas.numpy()
    areas_filtered = [area for area in areas if area[4] > confidence and area[-1] in coi]        
    return areas_filtered


def prep_nob(input_frames, target_frame, areas, pil2tensor, cfg, no_patch=False):
    '''
    No bactch allowed. Prepare (Patches + Frame) sets.
    i.e., [ [t0_p0, t1_p0, ..., ] ]
    Prepare inputs and targets to run generator forward.
    If no_path, do not crop and resize the entire frames.
    Else, do crop and stack patches.
    
    Should apply this function to one target_frame,
    since it generates several patches for detected areas (if exists).

    Inputs:
        areas: detected areas for one frame (array)
        pil2tensor: torchvision.transforms.ToTensor obejct
    Outputs:
        gen_inputs: stacked cropped patches in order to run in batch
                    (or) resized frames (if areas is None)
                        [B, C*nframe, resize_h, resize_w]
        gen_target: stacked cropped targets
                    (or) resized target (if areas is None)
                        [B, C, resize_h, resize_w]
    '''
    frames_all = input_frames + [target_frame]
    frames_all_resized = [data_utils.img2tensor(input_frame,
                            pil2tensor=pil2tensor,
                            resize_h=cfg.resize_h,
                            resize_w=cfg.resize_w,
                            symmetric=cfg.symmetric,
                            ) for input_frame in frames_all]
    # frame_inputs: [1, C*nframe, h, w], frame_target: [1, C, h, w]
    frame_inputs = torch.cat(frames_all_resized[:-1], dim=0).to(cfg.device).unsqueeze(0)
    frame_target = torch.as_tensor(frames_all_resized[-1], device=cfg.device).unsqueeze(0)
    if no_patch:
        gen_inputs = frame_inputs; gen_target = frame_target
    else:
        new_areas = crop_utils.get_resized_area(areas, max_w=cfg.max_w, max_h=cfg.max_h)
        # cropped_all = [ [t0_p0, t1_p0, ..., t4_p0], [t0_p1, ..., t4_p1], ... ]
        # t for timestamp and p for patch (area)
        cropped_all = crop_utils.crop_save(frames_all, new_areas, None, None, save=False)

        # Generate several patches in one frame at one batch
        gen_inputs = []; gen_target = []
        for one_area in cropped_all:
            # one_area_list[0]: [C, resize_h, resize_w]
            # len(one_area_list) = nframe + 1
            one_area_list = [data_utils.img2tensor(one_patch,
                                pil2tensor=pil2tensor,
                                resize_h=cfg.resize_h,
                                resize_w=cfg.resize_w,
                                symmetric=cfg.symmetric,
                                ) for one_patch in one_area]

            x_tmp = torch.cat(one_area_list[:-1], dim=0).to(cfg.device) # [C * nframe, h, w]
            y_tmp = torch.as_tensor(one_area_list[-1], device=cfg.device) # [C, h, w]
            gen_inputs.append(x_tmp); gen_target.append(y_tmp)
        # B: num of patches + 1 (1 for the whole frame)
        gen_inputs = torch.stack(gen_inputs, dim=0).contiguous().to(cfg.device) # [B + 1, C * nframe, h, w]
        gen_target = torch.stack(gen_target, dim=0).contiguous().to(cfg.device) # [B + 1, C, h, w]
        gen_inputs = torch.cat([gen_inputs, frame_inputs], dim=0)
        gen_target = torch.cat([gen_target, frame_target], dim=0)
        assert gen_inputs.size()[0] == len(cropped_all) + 1

    return gen_inputs, gen_target


def _calc_area(area):
    dw = area[2] - area[0]
    dh = area[3] - area[1]
    s = dw * dh
    return s


def infer_calc_area(areas, squash_func):
    sizes = [_calc_area(area) for area in areas]
    sizes = squash_func(sizes)
    mx = np.max(sizes)
    sizes_norm = np.array([mx/size for size in sizes])
    return sizes_norm, sizes


def step_train(inputs, model, losses,
                opts, schedulers,
                cfg, epoch, global_step=None, cur_iter=None):
    '''
    inputs: [B, C * nframe, H, W]
    '''
    
    opt_g = opts
    sch_g = schedulers

    b = inputs.shape[0]

    target = torch.cat([torch.Tensor([[0, 1]]*b), torch.Tensor([[1,0]]*b)], dim=0).cuda()
        
    ta_loss, pred_ta = timeArrow(inputs, target, model, losses[0], cfg, global_step)
    rot_loss, pred_rot = rotation(inputs, target, model, losses[2], cfg, global_step)
    irr_loss, pred_irr = irregularity(inputs, target, model, losses[1], cfg, global_step)

    print(pred_ta)
    # quit()
    # print(pred_rot)
    # print(pred_irr)
    

    loss_tot =  ta_loss + irr_loss + rot_loss*0.05
    
    opt_g.zero_grad()
    loss_tot.backward()
    opt_g.step()

    if sch_g is not None:
        sch_g.step()

    if cur_iter % 100 == 0:
        print(f'[Train-{epoch}-{cur_iter}] Total loss: {loss_tot.item():.2f} | ', 
        f'ta_loss: {ta_loss.item():.2f} | irr_loss: {irr_loss.item():.2f} | rot_loss: {rot_loss.item():.2f}')

    if global_step % cfg.verbose == 0:
        logger.info(f'[Train-{epoch}-{global_step}] Total loss: {loss_tot.item():.2f} |')

    return pred_ta, pred_irr, pred_rot


def one_direction(inputs, targets,
                    models, losses,
                    cfg, dicrection,
                    global_step=None):
    '''
    Return generator_loss, discriminator_loss, generated_frame
    '''
    coefs = cfg.lambdas
    nframe = cfg.nframe
    generator, discriminator, flownet = models
    adversarial_loss, discriminate_loss, \
        gradient_loss, intensity_loss, \
        flow_loss = losses

    # one direction
    # generator
    pred = generator(inputs)
    inte_l = intensity_loss(pred, targets)
    grad_l = gradient_loss(pred, targets)
    adv_l = adversarial_loss(discriminator(pred))

    # flownet
    input_last = inputs[:, (nframe-1)*3: nframe*3]
    if cfg.flownet == 'lite':
        gt_flow_input = torch.cat([input_last, targets], 1)
        pred_flow_input = torch.cat([input_last, pred], 1)
        # No need to train flow_net, use .detach() to cut off gradients.
        # input for lite: [-1, 1]
        flow_gt = flownet.batch_estimate(gt_flow_input, flownet).detach()
        flow_pred = flownet.batch_estimate(pred_flow_input, flownet).detach()
    else:
        gt_flow_input = torch.cat([input_last.unsqueeze(2), targets.unsqueeze(2)], 2)
        pred_flow_input = torch.cat([input_last.unsqueeze(2), pred.unsqueeze(2)], 2)
        # input for flownet2sd: [0, 255]
        gt_flow_input = (gt_flow_input+1)/2
        pred_flow_input = (pred_flow_input+1)/2
        flow_gt = (flownet(gt_flow_input * 255.) / 255.).detach()
        flow_pred = (flownet(pred_flow_input * 255.) / 255.).detach()
    flow_l = flow_loss(flow_pred, flow_gt)

    loss_gen = coefs[0] * inte_l + \
                coefs[1] * grad_l + \
                coefs[2] * adv_l + \
                coefs[3] * flow_l
    # discriminator
    loss_dis = discriminate_loss(discriminator(targets),
                                discriminator(pred.detach()))
    if global_step % cfg.verbose == 0:
        logger.info(f'[Train-{global_step}] {dicrection} | gen: {loss_gen.item():.2f} |')
        logger.info(f' | inte_l: {inte_l.item():.2f}, grad_l: {grad_l.item():.2f}, adv_l: {adv_l.item():.2f}, flow_l: {flow_l.item():.2f}')
        logger.info(f'[Train-{global_step}] {dicrection} | dis: {loss_dis.item():.2f} |')

    return loss_gen, loss_dis, pred


def timeArrow(inputs, target,
                    model, loss,
                    cfg, global_step=None):

    # inputs: t t+1 t+2 t+3 t+4 (b, t, c, w, h)
    # output: 
    inputs = inputs[:,:5]

    rev_inputs = inputs.flip([1])
    

    tot_inputs = torch.cat([inputs, rev_inputs], dim=0)
    
    ta_res = model(tot_inputs, 'ta', levels=['sequential', 'space'])
    
    ta_loss = loss(ta_res, target)

    if global_step % cfg.verbose == 0:
        logger.info(f'[Train-{global_step}] | ta_loss: {ta_loss.item():.2f}')

    return ta_loss, ta_res


def irregularity(inputs, target,
                    model, loss,
                    cfg, global_step=None):

    # inputs: t t+1 t+2 t+3 ... t+8  (b, t, c, w, h)
    # output: 

    diff = 4

    while diff < 7:
        indice = torch.Tensor(sorted(random.sample(range(0,8),5))).to(torch.int).cuda()
        diff = indice[-1]-indice[0]

    reg_inputs = inputs[:,:5]
    irr_inputs = inputs.index_select(1, indice)

    tot_inputs = torch.cat([reg_inputs, irr_inputs], dim=0)

    irr_res = model(tot_inputs, 'irr', levels=['sequential', 'space'])

    irr_loss = loss(irr_res, target)

    if global_step % cfg.verbose == 0:
        logger.info(f'[Train-{global_step}] | irr_loss: {irr_loss.item():.2f}')

    return irr_loss, irr_res

def rotation(inputs, target,
                    model, loss,
                    cfg, global_step=None):

    # inputs: t t+1 t+2 t+3 t+4 
    # output: 

    inputs = inputs[:,:5]

    rot_i = random.randint(1,3)

    rot_inputs = inputs.rot90(rot_i,[3,4])
    tot_inputs = torch.cat([inputs, rot_inputs], dim=0)
    rot_res = model(tot_inputs, 'rot', levels=['sequential', 'space'])

    rot_loss = loss(rot_res, target)

    if global_step % cfg.verbose == 0:
        logger.info(f'[Train-{global_step}] | ta_loss: {rot_loss.item():.2f}')

    return rot_loss, rot_res

def cal_acc(pred_ta, pred_irr, pred_rot):

    b = int(len(pred_ta)/2)

    target = torch.cat([torch.ones([b]), torch.zeros([b])]).cuda()
    
    ta_acc = torch.square(pred_ta.argmax(dim=1)-target).sum().detach()
    irr_acc = torch.square(pred_irr.argmax(dim=1)-target).sum().detach()
    rot_acc = torch.square(pred_rot.argmax(dim=1)-target).sum().detach()
    
    tot_acc = ta_acc + irr_acc + rot_acc

    return tot_acc,ta_acc , irr_acc , rot_acc, b*2