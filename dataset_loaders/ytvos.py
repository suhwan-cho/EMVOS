from torch.utils.data import Dataset
import os
from glob import glob
import torchvision as tv
import torchvision.transforms.functional as TF
from .transforms import *


class TrainYTVOS(Dataset):
    def __init__(self, root, output_size, clip_l, clip_n):
        self.root = os.path.join(root)
        self.output_size = output_size
        self.clip_l = clip_l
        self.clip_n = clip_n
        with open(os.path.join(root, 'ImageSets', 'train.txt'), 'r') as f:
            self.video_list = f.read().splitlines()
        self.to_tensor = tv.transforms.ToTensor()
        self.to_mask = LabelToLongTensor()

    def __len__(self):
        return self.clip_n

    def __getitem__(self, idx):
        video_name = random.choice(self.video_list)
        img_dir = os.path.join(self.root, 'train', 'JPEGImages', video_name)
        mask_dir = os.path.join(self.root, 'train', 'Annotations', video_name)
        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        # reverse video
        if random.random() > 0.5:
            img_list.reverse()
            mask_list.reverse()

        # get flip param
        h_flip = False
        if random.random() > 0.5:
            h_flip = True
        v_flip = False
        if random.random() > 0.5:
            v_flip = True

        # select training frames
        all_frames = list(range(len(img_list)))
        selected_frames = random.sample(all_frames, 1)
        for _ in range(self.clip_l - 1):
            if selected_frames[-1] + 1 > all_frames[-1]:
                selected_frames.append(selected_frames[-1])
            else:
                selected_frames.append(selected_frames[-1] + 1)

        # generate training snippets
        img_lst = []
        mask_lst = []
        for i, frame_id in enumerate(selected_frames):
            img = load_image_in_PIL(img_list[frame_id], 'RGB')
            mask = load_image_in_PIL(mask_list[frame_id], 'P')

            # resize to 480p
            W, H = img.size[0], img.size[1]
            if H > W:
                ratio = 480 / W
                img = img.resize((480, int(ratio * H)), Image.BICUBIC)
                mask = mask.resize((480, int(ratio * H)), Image.NEAREST)
            else:
                ratio = 480 / H
                img = img.resize((int(ratio * W), 480), Image.BICUBIC)
                mask = mask.resize((int(ratio * W), 480), Image.NEAREST)

            # joint flip
            if h_flip:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if v_flip:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            # joint balanced random crop
            if i == 0:
                for cnt in range(10):
                    y, x, h, w = random_crop_params(mask, scale=(0.8, 1.25))
                    temp_mask = self.to_mask(TF.resized_crop(mask, y, x, h, w, self.output_size, Image.NEAREST))

                    # select one object from reference frame
                    selected_id = 19971007
                    possible_obj_ids = temp_mask.unique().tolist()
                    if 0 in possible_obj_ids:
                        possible_obj_ids.remove(0)
                    if len(possible_obj_ids) > 0:
                        selected_id = random.choice(possible_obj_ids)
                    temp_mask[temp_mask != selected_id] = 0
                    temp_mask[temp_mask != 0] = 1

                    # ensure at least 256 FG pixels
                    if len(temp_mask[temp_mask != 0]) >= 256 or cnt == 9:
                        img_lst.append(self.to_tensor(TF.resized_crop(img, y, x, h, w, self.output_size, Image.BICUBIC)))
                        mask_lst.append(temp_mask)
                        break
            else:
                img_lst.append(self.to_tensor(TF.resized_crop(img, y, x, h, w, self.output_size, Image.BICUBIC)))
                temp_mask = self.to_mask(TF.resized_crop(mask, y, x, h, w, self.output_size, Image.NEAREST))
                temp_mask[temp_mask != selected_id] = 0
                temp_mask[temp_mask != 0] = 1
                mask_lst.append(temp_mask)
            imgs = torch.stack(img_lst, 0)
            masks = torch.stack(mask_lst, 0)
        return {'imgs': imgs, 'masks': masks}


class TestYTVOS(Dataset):
    def __init__(self, root):
        self.root = root
        self.init_data()

    def read_img(self, path):
        pic = Image.open(path)
        transform = tv.transforms.ToTensor()
        return transform(pic)

    def read_mask(self, path):
        pic = Image.open(path)
        transform = LabelToLongTensor()
        return transform(pic)

    def init_data(self):
        self.video_list = sorted(os.listdir(os.path.join(self.root, 'valid', 'Annotations')))
        print('--- YTVOS 2018 val loaded for testing ---')

    def get_snippet(self, video_name, frame_ids, val_frame_ids):
        img_path = os.path.join(self.root, 'valid_all_frames', 'JPEGImages', video_name)
        mask_path = os.path.join(self.root, 'valid', 'Annotations', video_name)
        imgs = torch.stack([self.read_img(os.path.join(img_path, '{:05d}.jpg'.format(i))) for i in frame_ids]).unsqueeze(0)
        given_masks = [self.read_mask(os.path.join(mask_path, '{:05d}.png'.format(i))).unsqueeze(0)
                       if i in sorted([int(os.path.splitext(file)[0]) for file in os.listdir(mask_path)]) else None for i in frame_ids]
        files = ['{:05d}.png'.format(i) for i in val_frame_ids]
        return {'imgs': imgs, 'given_masks': given_masks, 'files': files, 'val_frame_ids': val_frame_ids}

    def get_video(self, video_name):
        frame_ids = sorted([int(os.path.splitext(file)[0]) for file in os.listdir(os.path.join(self.root, 'valid_all_frames', 'JPEGImages', video_name))])
        val_frame_ids = sorted([int(os.path.splitext(file)[0]) for file in os.listdir(os.path.join(self.root, 'valid', 'JPEGImages', video_name))])
        min_frame_id = sorted([int(os.path.splitext(file)[0]) for file in os.listdir(os.path.join(self.root, 'valid', 'Annotations', video_name))])[0]
        frame_ids = [i for i in frame_ids if i >= min_frame_id]
        val_frame_ids = [i for i in val_frame_ids if i >= min_frame_id]
        yield self.get_snippet(video_name, frame_ids, val_frame_ids)

    def get_videos(self):
        for video_name in self.video_list:
            yield video_name, self.get_video(video_name)
