# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import itertools

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", default='data/vaw/annotations/vaw_train.json')
    parser.add_argument("--ann_idx", default='data/vaw/annotations/attribute_index.json')
    parser.add_argument("--add_freq", action='store_true')
    parser.add_argument("--r_thresh", type=int, default=10)
    parser.add_argument("--c_thresh", type=int, default=100)
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    idx_data = json.load(open(args.ann_idx, 'r'))
    cls_idx = {v:k for k,v in idx_data.items()}
    instance_counts = {v: 0 for k,v in idx_data.items()}
    image_counts = {v: 0 for k,v in idx_data.items()}
    for x in data:
        cat_id = list(itertools.chain(*x['pos_att_id']))

        for idx in cat_id:
            instance_counts[idx]+=1
        instance_id = list(set(cat_id))
        for idx in instance_id:
            image_counts[idx]+=1
    cat_info = []
    for k,v in cls_idx.items():
        cat_info.append({
            'id':k,
            'image_count':image_counts[k],
            'instance_count':instance_counts[k],
        })

    instance_num = sum([v for k,v in instance_counts.items()])
    image_num = sum([v for k,v in image_counts.items()])
    print(instance_num,image_num)
    import pdb; pdb.set_trace()
    out_path = '/vaw_train_cat_info.json'
    json.dump(cat_info, open('/'.join(args.ann.split('/')[:3])+out_path, 'w'))
        
    
