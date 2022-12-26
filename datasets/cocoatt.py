import torch
import torch.utils.data.dataset as dataset
import pickle
from PIL import Image
import numpy as np
# from sklearn.externals import joblib
import logging
from pathlib import Path
import os.path as osp
#from pycocotools.coco import COCO

class COCOAttributes(dataset.Dataset):
    def __init__(self, attributes_file, annotations_file, dataset_root,
                 transforms=None, target_transforms=None,
                 split='train2014', train=True,
                 n_attrs=204, crop_size=224):
        with open(attributes_file,'rb') as f:
            self.attributes_dataset = pickle.load(f,encoding='latin1')
            # self.attributes_dataset = joblib.load(osp.join(dataset_root,
            #                                            attributes_file))
        self.coco = COCO(annotations_file)
        self.dataset_root = osp.join(dataset_root,'images')

        self.transforms = transforms
        self.target_transforms = target_transforms

        self.split = split
        self.train = train
        self.n_attrs = n_attrs
        self.crop_size = crop_size

        self.data = []
        # get all attribute vectors for this split
        for patch_id, _ in self.attributes_dataset['ann_vecs'].items():
            if self.attributes_dataset['split'][patch_id] == split:
                self.data.append(patch_id)

        # list of attribute names
        self.attributes = sorted(
            self.attributes_dataset['attributes'], key=lambda x: x['id'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        patch_id = self.data[index]

        attrs = self.attributes_dataset['ann_vecs'][patch_id]
        attrs = (attrs > 0).astype(np.float)

        ann_id = self.attributes_dataset['patch_id_to_ann_id'][patch_id]
        # coco.loadImgs returns a list
        ann = self.coco.loadAnns(ann_id)[0]
        image = self.coco.loadImgs(ann['image_id'])[0]

        x, y, width, height = ann["bbox"]

        img = Image.open(osp.join(self.dataset_root, self.split,
                                  image["file_name"])).convert('RGB')

        # Crop out the object with context padding.
        img = get_image_crop(img, x, y, width, height, self.crop_size)

        # Transforms should be:
        # 1. Resize
        # 2. RandomHorizontalFlip
        # 3. Normalize
        # 4. Convert to Tensor (CxHxW)
        if self.transforms is not None:
            img = self.transforms(img)

        # Torch automatically converts numpy arrays to Tensors so target_transform can be None
        if self.target_transforms is not None:
            attrs = self.target_transforms(attrs)

        return img, attrs
def get_image_crop(img, x, y, width, height, crop_size=224, padding=16):
    """
    Get the image crop for the object specified in the COCO annotations.
    We crop in such a way that in the final resized image, there is `context padding` amount of image data around the object.
    This is the same as is used in RCNN to allow for additional image context.
    :param img: The image ndarray
    :param x: The x coordinate for the start of the bounding box
    :param y: The y coordinate for the start of the bounding box
    :param width: The width of the bounding box
    :param height: The height of the bounding box
    :param crop_size: The final size of the cropped image. Needed to calculate the amount of context padding.
    :param padding: The amount of context padding needed in the image.
    :return:
    """
    # Scale used to compute the new bbox for the image such that there is surrounding context.
    # The way it works is that we find what is the scaling factor between the crop and the crop without the padding
    # (which would be the original tight bounding box).
    # `crop_size` is the size of the crop with context padding.
    # The denominator is the size of the crop if we applied the same transform with the original tight bounding box.
    scale = crop_size / (crop_size - padding * 2)

    # Calculate semi-width and semi-height
    semi_width = width / 2
    semi_height = height / 2

    # Calculate the center of the crop
    centerx = x + semi_width
    centery = y + semi_height

    img_width, img_height = img.size

    # We get the crop using the semi- height and width from the center of the crop.
    # The semi- height and width are scaled accordingly.
    # We also ensure the numbers are valid
    upper = max(0, centery - (semi_height * scale))
    lower = min(img_height, centery + (semi_height * scale))
    left = max(0, centerx - (semi_width * scale))
    right = min(img_width, centerx + (semi_width * scale))

    crop_img = img.crop((left, upper, right, lower))

    if 0 in crop_img.size:
        print(img.size)
        print("lowx {0}\nlowy {1}\nhighx {2}\nhighy {3}".format(
            left, upper, right, lower))

    return crop_img

# Add color jitter to coco transforms
def make_vaw_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': ('train2014' , root / 'annotations' / 'instances_train2014.json'),
        'val': ('val2014' , root / 'annotations' / 'instances_val2014.json'),
    }
    attr_dir = 'data/coco_attributes/data/cocottributes_eccv_version.pkl'
    # CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.npy'
    # attribute_freq = root / 'annotations' / 'vaw_train_cat_info.json'
    # attribute_index = root / 'annotations' / 'attribute_index.json'
    
    split, anno_file = PATHS[image_set]
    is_train = True if image_set =='train' else False
    dataset = COCOAttributes(attr_dir, anno_file, root, split = split, train = is_train, 
                                )
    # if image_set == 'val' or image_set == 'test':
            
        # dataset.set_rare_attrs(attribute_freq)
        # dataset._valid_att_idxs(attribute_freq)
        
    return dataset