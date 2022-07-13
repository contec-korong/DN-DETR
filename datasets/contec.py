from math import pi
import cv2
import os

from pycocotools.coco import COCO
import torch
import torch.utils.data as data
import numpy as np


class ContecDetection(data.Dataset):
    def __init__(self, img_dir, ann_file, img_size):
        self.ann_file = COCO(ann_file)
        self.img_ids = self.ann_file.getImgIds()
        self.img_dir = img_dir
        self.input_size = img_size

    def __getitem__(self, index):
        """Returns an image and its ground-truth

        Output:Init type:
            - target: dict of multiple items
                - boxes: Tensor[num_box, 5]. \
                    Init type: cx,cy,w,h,angle. unnormalized data. unnormalized data.
                    Final type: cx,cy,w,h,angle. unnormalized data. normalized data.
                #- boxes: Tensor[num_box, 4]. \
                #    Init type: x0,y0,x1,y1. unnormalized data.
                #    Final type: cx,cy,w,h. normalized data.

        Args:
            index (int): Image ID

        Returns:
            img (torch): An image
            target (list[dict]): A Ground-truth
        """
        ids = self.img_ids[index]
        img = self.get_img(ids)

        targets = self.get_targets(ids)
        for k in targets.keys():
            targets[k] = torch.from_numpy(targets[k])

        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        return img, targets

    def __len__(self):
        return len(self.img_ids)

    def get_img(self, ids):
        """Read numpy image

        Args:
            ids (int): Image index

        Returns:
            img (ndarray): Image of form (H, W, C)
        """
        img_info = self.ann_file.imgs[ids]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        img = np.load(img_path)

        if isinstance(img, type(None)):
            raise FileNotFoundError(img_path)

        if img.shape[0] != self.input_size or img.shape[1] != self.input_size:
            raise NotImplementedError('Image resize is not implemented yet.')

        return img

    def get_targets(self, ids):
        """Read ground-truth rotated bounding boxes

        Args:
            ids (int): Image index

        Returns:
            dict:
                gt_bboxes (ndarray): Ground-truth bboxes of ids-th image of shape [n, 5]: (cx, cy, w, h, angle)
                gt_bboxes (ndarray): Ground-truth labels of gt_bboxes (normalized data [0, 1])
        """
        annotations = self.ann_file.imgToAnns[ids]

        gt_bboxes = []
        gt_labels = []

        for annotation in annotations:
            gt_bboxes.append(annotation['rbbox'][8:13])  # cx, cy, w, h, angle
            gt_labels.append(annotation['category_id'] - 1)  # First class should be 0

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)

            gt_bboxes[gt_bboxes < 0] = 0
            gt_bboxes[gt_bboxes >= self.input_size] = self.input_size - 1
            gt_bboxes[:, :4] = gt_bboxes[:, :4] / self.input_size   # Normalize cx, cy, w, h
            gt_bboxes[:, 4] = gt_bboxes[:, 4] / pi                  # Normalize angle
        else:
            gt_bboxes = np.zeros((0, 5), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        return {'boxes': gt_bboxes, 'labels': gt_labels}


def build(image_set, img_size):
    root = 'datasets'
    PATHS = {
        "train": [os.path.join(root, 'train', 'images'), os.path.join(root, 'train', 'annotations', 'train.json')],
        "test": [os.path.join(root, 'test', 'images'), os.path.join(root, 'test', 'annotations', 'test.json')]
    }

    img_dir, ann_file = PATHS[image_set]

    dataset = ContecDetection(img_dir, ann_file, img_size)

    return dataset


def draw(img, boxes, thr=0.1):
    """
    Convert numpy image to png file

    Args:
        img: Numpy arrays: (H, W, C)
        boxes([dict]): Model output. [{'scores': s, 'labels': l, 'boxes': b}]

    Returns:
        img: PNG image (H, W, C)
    """
    color = (255, 0, 0)
    thickness = 2

    # Normalize image [0, 255]
    for ch in range(img.shape[-1]):
        img[..., ch] = (img[..., ch] - np.min(img[..., ch])) / (np.max(img[..., ch]) - np.min(img[..., ch])) * 255

    img = np.int0(img)
    img = cv2.UMat(img)

    rboxes = []
    for i, box in enumerate(boxes):
        if box['scores'] > thr:
            rbox = box['boxes'].detach().cpu().numpy()
            rbox = ((rbox[0], rbox[1]), (rbox[2], rbox[3]), rbox[4]*np.pi/180)
            rbox = cv2.boxPoints(rbox)
            rbox = np.int0(rbox)  # Convert into integer values
            rboxes.append(rbox)

    drawn_img = cv2.drawContours(img, rboxes, -1, color, thickness)

    return drawn_img



