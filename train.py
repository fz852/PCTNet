import sys
import time
import os
import glob
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import torch
import warnings
import argparse
import numpy as np
from tqdm import tqdm
from model import PCTNet
from utils import torch_DataLoader_OASIS, dice_coef, JacboianDet, topology_change, generate_grid
import loss, utils
def train():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='param')
    parser.add_argument('--img_dir',
                        default="/public/wlj/datasets/Brain_MRI/affine_img_ordered/", type=str)
    parser.add_argument('--seg_dir',
                        default="/public/wlj/datasets/Brain_MRI/affine_seg_ordered/", type=str)
    parser.add_argument('--GPU_id', default='2', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--reproducible_seed', default=None)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--smooth_loss', default=0.02, type=float)
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id
    warnings.filterwarnings('ignore')
    dataset_img = glob.glob(args.img_dir + '*.nii.gz')
    dataset_img.sort(key=lambda x: int(x[54:-7]))
    dataset_seg = glob.glob(args.seg_dir + '*.nii.gz')
    dataset_seg.sort(key=lambda x: int(x[54:-7]))
    assert len(dataset_seg) == len(dataset_img), 'Image number != Segmentation number'

    training_loader = torch_DataLoader_OASIS(dataset_img, dataset_seg, 'train', args.batch_size, random_seed=args.reproducible_seed)
    testing_loader = torch_DataLoader_OASIS(dataset_img, dataset_seg, 'test', args.batch_size, random_seed=args.reproducible_seed)
    labels = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 60]
    grid1 = generate_grid([160, 192, 224])
    grid1 = torch.from_numpy(np.reshape(grid1, (1,) + grid1.shape)).cuda().float()
    best_dice = 0
    # model = model().cuda()
    model = PCTNet(
        in_chans=2,
        out_chans=5,
        depths=[2, 2, 4, 2],
        feat_size=[96, 192, 384, 768],
        channel_dim=[768, 384, 192, 96],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    )
    model.cuda()
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    criterion = torch.nn.MSELoss()
    criterions = [criterion]
    criterions += [loss.Grad3d(penalty='l2')]
    weights = [1, args.smooth_loss]
    current_iter = 0
    load_model = False
    if load_model is True:
        model_path = "stage_30000.pth"
        print("Loading weight: ", model_path)
        current_iter = 30000
        model.load_state_dict(torch.load(model_path))
    loss_all = utils.AverageMeter()


    for pair, moving_img, fixed_img, moving_seg, fixed_seg in tqdm(training_loader, desc='Current Alignment Progress'):


        model.train()
        pair = tuple(int(x[0]) for x in pair)
        moving_img = moving_img.unsqueeze(0).cuda()
        fixed_img = fixed_img.unsqueeze(0).cuda()
        input_img = torch.cat([moving_img, fixed_img], 1)
        time1_start = time.time()
        output = model(input_img)
        time1 = time.time() - time1_start
        reg_model = utils.register_model((160, 192, 224), 'nearest')
        reg_model.cuda()
        loss = 0
        loss_vals = []

        for n, loss_function in enumerate(criterions):
            curr_loss = loss_function(output[n], fixed_img) * weights[n]
            loss_vals.append(curr_loss)
            loss += curr_loss
        loss_all.update(loss.item(), fixed_img.numel())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del input_img
        del output
        # flip fixed and moving images
        loss = 0
        input_img = torch.cat([fixed_img, moving_img], dim=1)
        time2_start = time.time()
        output = model(input_img)
        time2 = time.time() - time1_start
        for n, loss_function in enumerate(criterions):
            curr_loss = loss_function(output[n], moving_img) * weights[n]
            loss_vals[n] += curr_loss
            loss += curr_loss
        loss_all.update(loss.item(), moving_img.numel())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model_dir = './experiments'
        print('\n' + str(pair[0]) + '->' + str(pair[1]) + '  loss {:.6f}, Img Sim: {:.6f}, Reg: {:.6f}, time: {:.6f}'.format(loss.item(), loss_vals[0].item() / 2,
                                                               loss_vals[1].item() / 2, (time1 + time2) / 2))
        if (current_iter % 2000 == 0 and current_iter != 0):
            modelname = model_dir + '/' + "stage_" + str(current_iter) + '.pth'
            torch.save(model.state_dict(), modelname)
            runtime = []
            jc = []
            TC = []
            so_dice = []
            re_dice = []
            for pair, moving_img, fixed_img, moving_seg, fixed_seg in tqdm(testing_loader, desc='Current testing progress'):
                with torch.no_grad():
                    moving_img = moving_img.unsqueeze(0).cuda()
                    fixed_img = fixed_img.unsqueeze(0).cuda()
                    moving_seg = moving_seg.unsqueeze(0).cuda()
                    fixed_seg = fixed_seg.unsqueeze(0).cuda()
                    x_in = torch.cat((moving_img, fixed_img), dim=1)
                    start_time = time.time()
                    output = model(x_in)
                    end_time = time.time()
                    X_Y = reg_model([moving_img.cuda().float(), output[1].cuda()]).data.cpu().numpy()[0, 0, :, :, :]
                    flow_norm = output[1].permute(0, 2, 3, 4, 1)
                    flow_jc = output[1]
                    jac_det = utils.jacobian_determinant_vxm(flow_jc.detach().cpu().numpy()[0, :, :, :, :])
                    jc.append(np.sum(jac_det <= 0) / np.prod((160, 192, 224)))
                    runtime.append(end_time - start_time)
                    X_Y1 = reg_model([moving_seg.cuda().float(), output[1].cuda()]).data.cpu().numpy()[0, 0, :, :, :]
                    F_X_Y_cpu = output[1].data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
                    TC.append(topology_change(moving_seg.data.cpu().numpy()[0, 0, :, :, :], X_Y1, labels))
                    so_dice.append(dice_coef(moving_seg.data.cpu().numpy()[0, 0, :, :, :], fixed_seg.data.cpu().numpy()[0, 0, :, :, :], labels))
                    re_dice.append(dice_coef(X_Y1, fixed_seg.data.cpu().numpy()[0, 0, :, :, :], labels))
                    sys.stdout.write(
                        "\r" + pair[0][0] + '->' + pair[1][0] + ' - Source_dice "{0:.9f}" - Regist_dice "{1:.9f}" - Jacc "{2:.9f}"  - TC "{3:.9f}" -Time "{4:.9f}"'.format(
                            so_dice[-1], re_dice[-1], jc[-1],  TC[-1], runtime[-1]))
            aver_runtime = np.mean(runtime)
            aver_jc = np.mean(jc)
            aver_TC = np.mean(TC)
            aver_so_dice = np.mean(so_dice)
            aver_re_dice = np.mean(re_dice)
            if aver_re_dice > best_dice:
                best_dice = aver_re_dice
                modelname = model_dir + '/' + "stage_" + str(current_iter) + "_best" + '.pth'
                torch.save(model.state_dict(), modelname)
            print(
                "============================================Final result of number " + str(current_iter) + "=================================================")
            print(
                "\r" + 'Source_dice "{0:.9f}" - Regist_dice "{1:.9f}" - Jacc "{2:.9f}"  - TC "{3:.9f}" -Time "{4:.9f}"'.format(
                    aver_so_dice, aver_re_dice, aver_jc,  aver_TC, aver_runtime))
        current_iter += 1
