import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_ann', default='data/coco/zero-shot/instances_train2017.json')
    parser.add_argument('--att_index_ann', default='data/vaw/annotations/attribute_index.json')
    parser.add_argument('--train_part1', default='data/vaw/annotations/train_part1.json')
    parser.add_argument('--train_part2', default='data/vaw/annotations/train_part2.json')
    parser.add_argument('--val', default='data/vaw/annotations/val.json')
    parser.add_argument('--test', default='data/vaw/annotations/test.json')
    
    parser.add_argument('--vg_img_data', default='data/vaw/annotations/image_data.json')
    parser.add_argument('--out_path', default='data/vaw/annotations/vaw.json')
    args = parser.parse_args()

    attribute = json.load(open(args.att_index_ann,'r'))
    coco = json.load(open(args.coco_ann,'r'))
    vg_image= json.load(open(args.vg_img_data,'r'))
    split_dict = {'train': (json.load(open(args.train_part1,'r')),json.load(open(args.train_part2,'r'))),
                    'val':(json.load(open(args.val,'r')),),
                    'test': (json.load(open(args.test,'r')),),
                }
    split = ['train','val','test']

    coco_object_valid = [i['name'] for i in coco['categories']]
    coco_categories = coco['categories']
    
    # img_ids = [i['image_id'] for i in image]
    def process(items):
        imgs = {}
        i=items['id']
        data_=split_dict[items['type']]
        img_dict = []
        for dd in data_:
            img_dict.extend([d for d in dd if int(d['image_id'])==i])
        # img_dict = [d for d in data_ if int(d['image_id'])==i]
        file_name = next(im['url'] for im in vg_image if i == im['image_id'])
        imgs.update({'file_name':file_name,'image_id':i})
        bboxes = []
        pos_att_id,neg_att_id = [],[]
        category_id = []
        for id in img_dict:
            # import pdb;pdb.set_trace()
            # obj_id = next((item['id'] for item in coco_categories if item["name"] == id['object_name']),None)
            # if obj_id is None:
            #     continue
            category_id.append(1)
            bboxes.append(id['instance_bbox'])
            pos_att_id.append([attribute[pos_att] for pos_att in id['positive_attributes']])
            neg_att_id.append([attribute[neg_att] for neg_att in id['negative_attributes']])
        assert len(category_id)==len(bboxes)==len(pos_att_id)==len(neg_att_id)
        if len(category_id)!=0:
            imgs.update(
                {'boxes':bboxes,
                'category_id':category_id,
                'pos_att_id':pos_att_id,
                'neg_att_id':neg_att_id,

                }
            )
            return imgs
        else:
            return None
    for sp in split:
        data_ = split_dict[sp]
        
        images = {}
        anns=[]
        k=0
        img_ids = list(set([int(ann['image_id']) for dd in data_ for ann in dd]))
        items = [{'id':i,'type':sp} for i in img_ids]

        with Pool(32) as p:
            images = list(tqdm(p.imap(process, items), total=len(items)))
            print('DONE')
        json.dump(list(filter(None,images)), open(args.out_path[:-5]+'_orig_{}.json'.format(sp), 'w'))