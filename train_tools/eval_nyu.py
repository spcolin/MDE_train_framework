import torch
import torch.backends.cudnn as cudnn

import os, sys
import argparse
import numpy as np
from tqdm import tqdm

from utils import post_process_depth, flip_lr, compute_errors
from networks.NewCRFDepth import NewCRFDepth
from dataloaders.NYU_depth_dataset import NYU_DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='a training framework for monocular depth estimation')
parser.add_argument('--model_name',type=str,default='MDE_model',help='model name')

# Dataset
parser.add_argument('--dataset',type=str, default='nyu',help='dataset to train on, kitti or nyu')
parser.add_argument('--data_root',type=str,help='path to the data')
parser.add_argument('--filenames_file',type=str,help='path to the filenames text file')
parser.add_argument('--target_height',type=int,default=480,help='input height to model')
parser.add_argument('--target_width',type=int,default=640,help='input width to model')
parser.add_argument('--keep_border',type=bool,default=True,help='whether crop the blank area of data in NYU')
parser.add_argument('--random_crop', type=bool,default=False,help='if set, will perform random crop and resize to target size for augmentation')
parser.add_argument('--random_rotate',type=bool,default=True,help='if set, will perform random rotation for augmentation' )
parser.add_argument('--rotate_degree',type=float,default=2.5,help='random rotation maximum degree')
parser.add_argument('--depth_scale',type=float,default=1000.0,help='scale factor to keep the value of ground truth depth map between 0~10 during training ')
parser.add_argument('--random_flip',type=bool,default=True,help='if set, will perform random flip for augmentation')
parser.add_argument('--color_aug',type=bool,default=True,help='if set, will perform color jitter for augmentation')
parser.add_argument('--max_depth',type=float,default=10.0,help='maximum depth in estimation')

# Eval
parser.add_argument('--encoder',type=str,default='large07',help='type of encoder, base07, large07')
parser.add_argument('--checkpoint_path',type=str,default='',help='path to a checkpoint to load')
parser.add_argument('--data_root_eval',type=str,help='path to the data for online evaluation')
parser.add_argument('--filenames_file_eval',type=str, help='path to the filenames text file for online evaluation')
parser.add_argument('--min_depth_eval',type=float, default=1e-3, help='minimum depth for evaluation')
parser.add_argument('--max_depth_eval',type=float, default=10, help='maximum depth for evaluation')
parser.add_argument('--eigen_crop',type=bool,default=True,help='if set, crops according to Eigen NIPS14')

parser.add_argument('--project_path',type=str,default='/home/colin/code/depth_train_framework/v4/')


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

####################### train args #########################
args.model_name ='depth_model_nyu'
args.data_root_eval ='/home/colin/datasets/official_splits/test/'
args.filenames_file_eval = args.project_path+'data_splits/nyudepthv2_test_files_with_gt.txt'

args.checkpoint_path='/home/colin/pretrained_model/model_nyu.ckpt'
######################## args end ###########################



# if args.dataset == 'kitti' or args.dataset == 'nyu':
#     from dataloaders.dataloader import NewDataLoader
# elif args.dataset == 'kittipred':
#     from dataloaders.dataloader_kittipred import NewDataLoader


def eval(model, dataloader_eval, post_process=False):
    eval_measures = torch.zeros(10).cuda()
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue

            pred_depth = model(image)
            if post_process:
                image_flipped = flip_lr(image)
                pred_depth_flipped = model(image_flipped)
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        # if args.do_kb_crop:
        #     height, width = gt_depth.shape
        #     top_margin = int(height - 352)
        #     left_margin = int((width - 1216) / 2)
        #     pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
        #     pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
        #     pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.eigen_crop:
            eval_mask = np.zeros(valid_mask.shape)

            if args.eigen_crop:
               
                eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda()
        eval_measures[9] += 1

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                    'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                    'd3'))
    for i in range(8):
        print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.4f}'.format(eval_measures_cpu[8]))
    return eval_measures_cpu


def main_worker(args):

    # CRF model
    model = NewCRFDepth(version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=None)
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    model = torch.nn.DataParallel(model)
    model.cuda()

    print("== Model Initialized")

    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
            del checkpoint
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))

    cudnn.benchmark = True

    dataloader_eval = NYU_DataLoader(args, 'online_eval')

    # ===== Evaluation ======
    model.eval()
    with torch.no_grad():
        eval_measures = eval(model, dataloader_eval, post_process=True)


def main():
    torch.cuda.empty_cache()
    args.distributed = False
    # ngpus_per_node = torch.cuda.device_count()
    # if ngpus_per_node > 1:
    #     print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
    #     return -1
    
    main_worker(args)


if __name__ == '__main__':
    main()
