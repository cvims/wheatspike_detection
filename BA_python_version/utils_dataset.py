import torch
from torchvision import transforms as T
import albumentations as A
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa


def get_mean_std(loader):

    '''
    mean: tensor([0.2260, 0.2390, 0.1737])
    std: tensor([0.1723, 0.1773, 0.1578])
    '''

    """
    mean_all: tensor([0.2133, 0.2256, 0.1655])
    mean_all: tensor([0.1672, 0.1721, 0.1526])
    """

    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for i, (images, _) in enumerate(loader):
        image_list = []
        for image in images:
            image_list.append(get_transform_torch_img(get_transform_imgaug_img(image))/255)
        images = (image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], image_list[5], image_list[6], image_list[7])
        images = torch.stack(list(images), dim=0)
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
        print(i)

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std


# extract coco-informations
def extract_coco_annotations(file_path, idx):
    coco_annotation = COCO(annotation_file=file_path)
    
    category_ids = coco_annotation.getCatIds()
    categories = coco_annotation.loadCats(category_ids)
    category_names = [cat["name"] for cat in categories]
    query_id = category_ids[0]

    img_ids = coco_annotation.getImgIds(catIds=[query_id])
    img_id = img_ids[idx]

    ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco_annotation.loadAnns(ann_ids)
    mask = coco_annotation.annToMask(anns[0])

    new_mask = torch.zeros(size=(len(anns), mask.shape[0], mask.shape[1]))

    for i in range(1, len(anns)):
        new_mask[i] = new_mask[i] + coco_annotation.annToMask(anns[i])

    return anns, category_names, new_mask


# albumentation-composition of transformations
def get_transform_albumentation(train):
    transform = []
    transform.append(A.Rotate(limit=[39.925, 39.925], always_apply=True, crop_border=True))
    if train:
        transform.append(A.Affine(rotate=[-15, 15], p=0.5))
        transform.append(A.RandomCrop(height=164, width=164, p=0.5))
        transform.append(A.BBoxSafeRandomCrop(erosion_rate=0.2, p=0.5)) 
        transform.append(A.Flip(p=0.5))
        transform.append(A.RandomBrightnessContrast(p=0.2))
        transform.append(A.OneOf([
                            A.PixelDropout(p=0.6),
                            A.Perspective(p=0.4)
                            ], p=1))
    return A.Compose(transform, bbox_params=A.BboxParams(format='coco', label_fields=["class_labels"]))


# augments image
def get_transform_imgaug_img(img):
    seq = iaa.Sequential([
        iaa.Affine(rotate=-39.925),
        iaa.Resize({"height": 392, "width": 392}),
        iaa.CenterCropToFixedSize(width=280, height=280)
        ])
    numpy_img = np.array(img)
    image_aug= seq(image=numpy_img)
    image_aug = Image.fromarray(image_aug)
    return image_aug


    # transforms image from PIL to torch
def get_transform_torch_img(img):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.uint8))
    t = T.Compose(transforms)
    img = t(img)
    return img
