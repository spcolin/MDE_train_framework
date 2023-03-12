from torch.utils.data import Dataset
from PIL import Image
import os,random
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from utils import DistributedSamplerNoEvenlyDivisible

class NYU_DataLoader(object):
    def __init__(self, args,mode):
        
        if mode == 'train':

            self.samples = NYU_Depth(dataset_root=args.data_root,
                                              file_path=args.filenames_file,
                target_height=args.target_height,target_width=args.target_width,
                mode=mode,keep_border=args.keep_border,random_crop=args.random_crop,
                random_rotate=args.random_rotate,rotate_degree=args.rotate_degree,
                depth_scale=args.depth_scale,random_flip=args.random_flip,
                color_aug=args.color_aug)
            
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

            self.samples = NYU_Depth(dataset_root=args.data_root_eval,
                                            file_path=args.filenames_file_eval,
                target_height=args.target_height,target_width=args.target_width,
                mode=mode,keep_border=args.keep_border,random_crop=args.random_crop,
                random_rotate=args.random_rotate,rotate_degree=args.rotate_degree,
                depth_scale=args.depth_scale,random_flip=args.random_flip,
                color_aug=args.color_aug)
            
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
        
class NYU_Depth(Dataset):

    def __init__(self,dataset_root,file_path,target_height=480,target_width=640,
                mode='train',keep_border=True,random_crop=True,random_rotate=True,
                rotate_degree=2.5,depth_scale=1000.0,random_flip=True,color_aug=True):
        super().__init__()

        """
        dataset_root:the path placing NYU depth dataset
        file_path:training files containing the path of coupled rgb and depth map
        dataset_root+file_path locates the training or eval data
        target_height & target_width:the height and width of input tensor that finally
        get passed through network

        data augmentation option:
            keep_border: whether keep the blank border area coarse by registration between rgb and depth map
            random_crop: ...
            random_rotate: ...
            rotate_degree: the degree used for random_ratote option
            depth_scale: the ground truth depth map used for loss computation are to scaled into the scale between 0~max_depth (meter)
            randon_flip: ...
            color_aug: whether do data augmentation about intensity such as brightness
        """

        self.dataset_root=dataset_root
        self.target_height=target_height
        self.target_width=target_width
        self.mode=mode
        self.keep_border=keep_border
        self.random_crop=random_crop
        self.random_rotate=random_rotate
        self.rotate_degree=rotate_degree
        self.depth_scale=depth_scale
        self.random_flip=random_flip
        self.color_aug=color_aug
        
        file=open(file_path)

        self.files=file.readlines()

        file.close()


    def __len__(self):

        return len(self.files)

    def __getitem__(self,index):

        file_path=self.files[index]
        file_path=file_path.strip('\n')
        file_path=file_path.split(' ')

        rgb_path=file_path[0]
        depth_path=file_path[1]
        
        rgb_path=os.path.join(self.dataset_root,rgb_path)
        depth_path=os.path.join(self.dataset_root,depth_path)

        rgb=Image.open(rgb_path)
        depth=Image.open(depth_path)

        if self.mode=='train':

            if self.keep_border:
                depth = np.array(depth)
                valid_mask = np.zeros_like(depth)
                valid_mask[45:472, 43:608] = 1
                depth[valid_mask==0] = 0
                depth = Image.fromarray(depth)
            else:
                rgb = rgb.crop((43, 45, 608, 472))
                depth = depth.crop((43, 45, 608, 472))

            if self.random_crop:
                rgb,depth=self.crop(rgb,depth)
            
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

            return {'image': rgb_tensor, 'depth': depth_tensor}


        else:

            to_tensor_and_norm=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            rgb_tensor=to_tensor_and_norm(rgb)
            depth_tensor=transforms.ToTensor()(depth)

            depth_tensor=depth_tensor/self.depth_scale

            rgb_tensor=torch.nn.functional.interpolate(rgb_tensor.unsqueeze(1),size=(self.target_height,self.target_width),mode='bilinear').squeeze(1)

            return {'image': rgb_tensor, 'depth': depth_tensor,'has_valid_depth': True}

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

    def crop(self, rgb, depth):

        full_width,full_height=rgb.size

        height_scale=random.uniform(0.8,1)
        width_scale=random.uniform(0.8,1)

        height=int(full_height*height_scale)
        width=int(full_width*width_scale)

        x = random.randint(0, full_width - width)
        y = random.randint(0, full_height - height)

        rgb=rgb.crop((x,y,x+width,y+height))
        depth=depth.crop((x,y,x+width,y+height))

        return rgb,depth

    def augment_image(self, rgb):
        
        color_transform=transforms.ColorJitter(brightness=0.2,contrast=0.1,saturation=0.1,hue=0.1)

        rgb=color_transform(rgb)

        return rgb




