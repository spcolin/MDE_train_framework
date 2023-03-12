import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import numpy as np
from PIL import Image
import os
import random


class KITTI_DataLoader(object):
    def __init__(self, args,mode):
        
        if mode == 'train':

            self.samples = KITTI_Depth(data_path=args.data_path,gt_path=args.gt_path,
                file_path=args.filenames_file,
                 target_height=args.target_height,target_width=args.target_width,
                mode=mode,random_crop=args.random_crop,random_rotate=args.random_rotate,
                rotate_degree=args.rotate_degree,depth_scale=args.depth_scale,
                random_flip=args.random_flip,color_aug=args.color_aug,use_right=args.use_right,
                do_kb_crop=args.do_kb_crop)
            
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.samples)
            else:
                self.train_sampler = None
    
            self.data = DataLoader(self.samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        else:

            self.samples = KITTI_Depth(data_path=args.data_path,gt_path=args.gt_path,
                file_path=args.filenames_file_eval,
                 target_height=args.target_height,target_width=args.target_width,
                mode=mode,random_crop=args.random_crop,random_rotate=args.random_rotate,
                rotate_degree=args.rotate_degree,depth_scale=args.depth_scale,
                random_flip=args.random_flip,color_aug=args.color_aug,use_right=args.use_right,
                do_kb_crop=args.do_kb_crop)
            
            # if args.distributed:
            #     # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
            #     self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.samples, shuffle=False)
            # else:
            #     self.eval_sampler = None
            
            self.data = DataLoader(self.samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=None)
            

class KITTI_Depth(Dataset):
    def __init__(self,data_path,gt_path,file_path,
                 target_height=352,target_width=704,
                mode='train',random_crop=False,random_rotate=True,
                rotate_degree=1.0,depth_scale=256.0,random_flip=True,
                color_aug=True,use_right=True,do_kb_crop=True):
        
        """
        data_path:the path placing unzipped kitti raw
        gt_path:the path placing kitti depth annotated data
        file_path:training files containing the path of coupled rgb and depth map
        target_height & target_width:the height and width of input tensor that finally
        get passed through network

        data augmentation option:
            random_crop: ...
            random_rotate: ...
            rotate_degree: the degree used for random_ratote option
            depth_scale: the ground truth depth map used for loss computation are to scaled into the scale between 0~max_depth (meter)
            randon_flip: ...
            color_aug: whether do data augmentation about intensity such as brightness
            use_right: kitti raw are composed with binocular stereo images, whether to randomly choose the training data from right camera
            keep_border: whether to drop the top area of image where lidar labels are missing

        """

        self.data_path=data_path
        self.gt_path=gt_path

        self.target_height=target_height
        self.target_width=target_width
        self.mode=mode
        self.random_crop=random_crop
        self.random_rotate=random_rotate
        self.rotate_degree=rotate_degree
        self.depth_scale=depth_scale
        self.random_flip=random_flip
        self.color_aug=color_aug
        self.use_right=use_right
        self.do_kb_crop=do_kb_crop
        
        file=open(file_path)

        self.files=file.readlines()

        file.close()


    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):

        sample_path = self.files[idx]

        if self.mode=='train':

            rgb_file = sample_path.split()[0]
            depth_file = sample_path.split()[1]

            if self.use_right and (random.random() > 0.5):
                rgb_file.replace('image_02', 'image_03')
                depth_file.replace('image_02', 'image_03')
                
            image_path = os.path.join(self.data_path, rgb_file)
            depth_path = os.path.join(self.gt_path, depth_file)
        
            rgb = Image.open(image_path)
            depth = Image.open(depth_path)

            if self.do_kb_crop:
                height = rgb.height
                width = rgb.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth = depth.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                rgb = rgb.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            
            # if self.random_crop:
            #     rgb,depth=self.crop_resize(rgb,depth,self.target_height,self.target_width)

            if self.random_rotate:
                random_angle = (random.random() - 0.5) * 2 * self.rotate_degree
                rgb = self.rotate_image(rgb, random_angle)
                depth = self.rotate_image(depth, random_angle, flag=Image.NEAREST)

            if self.random_flip:
                rgb,depth=self.flip_image(rgb,depth)

            if self.color_aug:
                rgb=self.augment_image(rgb)

            to_tensor_and_norm=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            rgb_tensor=to_tensor_and_norm(rgb)
            depth_tensor=transforms.ToTensor()(depth)

            depth_tensor=depth_tensor/self.depth_scale

            rgb_tensor=torch.nn.functional.interpolate(rgb_tensor.unsqueeze(1),size=(self.target_height,self.target_width),mode='bilinear').squeeze(1)
            depth_tensor=torch.nn.functional.interpolate(depth_tensor.unsqueeze(1),size=(self.target_height,self.target_width),mode='bilinear').squeeze(1)

            sample = {'image': rgb_tensor, 'depth': depth_tensor}

            return sample

        else:

            image_path = os.path.join(self.data_path,sample_path.split()[0])
            depth_path = os.path.join(self.gt_path,sample_path.split()[1])
            
            rgb=Image.open(image_path)

            has_valid_depth = False

            try:
                depth = Image.open(depth_path)
                has_valid_depth = True
            except IOError:
                depth = False

            if self.do_kb_crop:
                height = rgb.height
                width = rgb.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                rgb = rgb.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                
                if has_valid_depth:
                    depth = depth.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            to_tensor_and_norm=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            rgb_tensor=to_tensor_and_norm(rgb)

            if has_valid_depth:

                depth_tensor=transforms.ToTensor()(depth)

                depth_tensor=depth_tensor/self.depth_scale

            else:

                depth_tensor=torch.zeros(rgb_tensor.shape)

            rgb_tensor=torch.nn.functional.interpolate(rgb_tensor.unsqueeze(1),size=(self.target_height,self.target_width),mode='bilinear').squeeze(1)

            sample = {'image': rgb_tensor, 'depth': depth_tensor,'has_valid_depth': has_valid_depth}

            return sample
            


    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def flip_image(self,rgb,depth):

        do_flip = random.random()
        if do_flip > 0.5:
            # print('flip image')
            rgb=rgb.transpose(Image.FLIP_LEFT_RIGHT)
            depth=depth.transpose(Image.FLIP_LEFT_RIGHT)

        return rgb,depth

    def crop_resize(self, rgb, depth, target_height, target_width):

        full_width,full_height=rgb.size

        height_scale=random.uniform(0.8,1)
        width_scale=random.uniform(0.8,1)

        height=int(full_height*height_scale)
        width=int(full_width*width_scale)

        x = random.randint(0, full_width - width)
        y = random.randint(0, full_height - height)

        rgb=rgb.crop((x,y,x+width,y+height))
        depth=depth.crop((x,y,x+width,y+height))

        rgb=rgb.resize((target_width,target_height),Image.ANTIALIAS)
        depth=depth.resize((target_width,target_height),Image.ANTIALIAS)

        return rgb,depth

    def augment_image(self, rgb):
        
        color_transform=transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.1)

        rgb=color_transform(rgb)

        return rgb




