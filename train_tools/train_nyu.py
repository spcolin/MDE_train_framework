import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import os, sys, time
from telnetlib import IP
import argparse
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

from utils import post_process_depth, flip_lr, silog_loss, compute_errors, eval_metrics, \
                       block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args
from networks.NewCRFDepth import NewCRFDepth
from dataloaders.NYU_depth_dataset import NYU_DataLoader
import random

def random_seed_fix(seed=369):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark=False
    # torch.backends.cudnn.deterministic=True

random_seed_fix()


parser = argparse.ArgumentParser(description='a training framework for monocular depth estimation')

parser.add_argument('--mode',type=str,default='train',help='train or eval')
parser.add_argument('--model_name',type=str,default='MDE_nyu',help='model name')

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

# Log and save
parser.add_argument('--log_directory',type=str, default='',help='directory to save checkpoints and summaries')
parser.add_argument('--log_freq',type=int,default=100,help='Logging frequency in global steps')
parser.add_argument('--save_freq',type=int,default=1000,help='Checkpoint saving frequency in global steps')

# Training
parser.add_argument('--adam_eps',type=float,default=1e-6,help='epsilon in Adam optimizer')
parser.add_argument('--batch_size',type=int,default=4,help='batch size')
parser.add_argument('--num_epochs',type=int,default=50,help='number of epochs')
parser.add_argument('--learning_rate',type=float,default=1e-4,help='initial learning rate')
parser.add_argument('--end_learning_rate',type=float,default=-1,help='end learning rate')
parser.add_argument('--variance_focus',type=float,default=0.85,help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error')

# model loading & resume training
parser.add_argument('--encoder',type=str,default='large07',help='type of encoder, base07, large07')
parser.add_argument('--pretrain',type=str,default=None,help='path of pretrained encoder')
parser.add_argument('--checkpoint_path',type=str, default='',help='path to a checkpoint to load')
parser.add_argument('--retrain',type=bool,default=False,help='if used with checkpoint_path, will restart training from step zero')

# Multi-gpu training
parser.add_argument('--num_threads',type=int,default=1,help='number of threads to use for data loading')
parser.add_argument('--world_size',type=int,default=1,help='number of nodes for distributed training')
parser.add_argument('--rank',type=int,default=0,help='node rank for distributed training')
parser.add_argument('--dist_url',type=str,default='tcp://127.0.0.1:1234',help='url used to set up distributed training')
parser.add_argument('--dist_backend',type=str,default='nccl',help='distributed backend')
parser.add_argument('--gpu',type=int,default=None,help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed',type=bool,default=True,help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training')
# Online eval
parser.add_argument('--do_online_eval',type=bool,default=True,help='if set, perform online eval in every eval_freq steps')
parser.add_argument('--data_root_eval',type=str,help='path to the data for online evaluation')
parser.add_argument('--filenames_file_eval',type=str,   help='path to the filenames text file for online evaluation')
parser.add_argument('--min_depth_eval',type=float, default=1e-3, help='minimum depth for evaluation')
parser.add_argument('--max_depth_eval',type=float, default=10, help='maximum depth for evaluation')
parser.add_argument('--eigen_crop',type=bool,default=True,help='if set, crops according to Eigen NIPS14')
parser.add_argument('--eval_freq',type=int, default=500,help='Online evaluation frequency in global steps')
parser.add_argument('--eval_summary_directory',type=str, default='',help='output directory for eval summary, if empty outputs to checkpoint folder')

parser.add_argument('--project_path',type=str,default='/home/colin/code/depth_train_framework/v4/')
parser.add_argument('--one_iter',default=False)


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

####################### train args #########################
args.project_path='/home/colin/code/depth_train_framework/v4/'
args.pretrain ='/home/colin/pretrained_model/swin_large_patch4_window7_224_22k.pth'
args.data_root ='/home/colin/datasets/sync/'
args.filenames_file= args.project_path+'data_splits/nyudepthv2_train_files_with_gt_dense.txt'
args.num_threads =1
args.log_directory=os.path.join(args.project_path,'logs')
args.multiprocessing_distributed=True
args.dist_url= 'tcp://127.0.0.1:3456'

args.data_root_eval ='/home/colin/datasets/official_splits/test/'
args.filenames_file_eval = args.project_path+'data_splits/nyudepthv2_test_files_with_gt.txt'

args.batch_size=2
args.eval_freq=1
# args.one_iter=True
# args.target_height=224
# args.target_width=224

######################## args end ###########################



def online_eval(model, dataloader_eval, gpu, ngpus, post_process=False):
    eval_measures = torch.zeros(10).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():

            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
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

            pred_shape=pred_depth.shape[2:]

            if pred_shape[0]!=480 or pred_shape[1]!=640:
                pred_depth=torch.nn.functional.interpolate(pred_depth,size=(480,640),mode='bilinear')

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

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

        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1

    # if args.multiprocessing_distributed:
    #     group = dist.new_group([i for i in range(ngpus)])
    #     dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
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

    return None


def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu  # the gpu index for parallel training

    if args.gpu is not None:
        print("== Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)


    # NeWCRFs model
    model = NewCRFDepth(version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=args.pretrain)
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.distributed:
        print("== Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("== Model Initialized")

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    # Training parameters
    optimizer = torch.optim.Adam([{'params': model.module.parameters()}],
                                lr=args.learning_rate)

    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if not args.retrain:
                try:
                    global_step = checkpoint['global_step']
                    best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                    best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                    best_eval_steps = checkpoint['best_eval_steps']
                except KeyError:
                    print("Could not load values for online evaluation")

            print("== Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True
        del checkpoint

    dataloader = NYU_DataLoader(args,'train')
    dataloader_eval = NYU_DataLoader(args, 'online_eval')


    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, args.model_name, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    silog_criterion = silog_loss(variance_focus=args.variance_focus)

    start_time = time.time()
    duration = 0

    # num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.01 * args.learning_rate

    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("== Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            optimizer.zero_grad()
            before_op_time = time.time()

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))

            depth_est = model(image)

            if args.dataset == 'nyu':
                mask = depth_gt > 0.1
            else:
                mask = depth_gt > 1.0

            loss = silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
            loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('silog_loss', loss, global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step)
                    writer.add_scalar('var average', var_sum.item()/var_cnt, global_step)
                    # depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
                    # for i in range(num_log_images):
                    #     writer.add_image('depth_gt/image/{}'.format(i), normalize_result(1/depth_gt[i, :, :, :].data), global_step)
                    #     writer.add_image('depth_est/image/{}'.format(i), normalize_result(1/depth_est[i, :, :, :].data), global_step)
                    #     writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
                    writer.flush()

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded and (not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)):
                time.sleep(0.1)
                model.eval()
                with torch.no_grad():
                    eval_measures = online_eval(model, dataloader_eval, gpu, ngpus_per_node, post_process=True)
                if eval_measures is not None:
                    for i in range(9):
                        eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            # model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            model_save_name = '/model_best_{}.pth'.format(eval_metrics[i])
                            print('New best for {}={}. Saving model: {}'.format(eval_metrics[i],measure, model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': model.state_dict(),
                                          'optimizer': optimizer.state_dict(),
                                          'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                          'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                          'best_eval_steps': best_eval_steps
                                          }
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                    eval_summary_writer.flush()
                model.train()
                block_print()
                enable_print()

            model_just_loaded = False
            global_step += 1

            if args.one_iter:
                break

        epoch += 1

        if args.one_iter:
            break
       
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer.close()
        if args.do_online_eval:
            eval_summary_writer.close()


def main():

    if args.mode != 'train':
        print('train.py is only for training.')
        return -1
    
    if not os.path.exists(args.log_directory):
        command='mkdir '+ args.log_directory
        os.system(command)
        print('training logs will be saved at:',args.log_directory)

    command = 'mkdir ' + os.path.join(args.log_directory, args.model_name)
    os.system(command)

    # save_files = False
    # if save_files:
    #     aux_out_path = os.path.join(args.log_directory, args.model_name)
    #     networks_savepath = os.path.join(aux_out_path, 'networks')
    #     dataloaders_savepath = os.path.join(aux_out_path, 'dataloaders')
    #     command = 'cp train_tools/train_nyu.py ' + aux_out_path
    #     os.system(command)
    #     command = 'mkdir -p ' + networks_savepath + ' && cp train_tools/networks/*.py ' + networks_savepath
    #     os.system(command)
    #     command = 'mkdir -p ' + dataloaders_savepath + ' && cp train_tools/dataloaders/*.py ' + dataloaders_savepath
    #     os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed  #True

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))
    

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
