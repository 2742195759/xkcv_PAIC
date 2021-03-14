"""
    首先去重复
    按照iid分类，然后选择5%作为testset, 5%作为valset。
    对每个val和test测试集，每个image,选择一个test集合存在的uid作为备选Caption任务就是拟合这个Caption
"""

import pandas as pd
import numpy as np

dataset_path = '../data/dataset_final.csv'
######################
######################

ds = pd.read_csv(dataset_path)
ds = ds[~ds['review'].isna()]

user_num = len(ds['uid'].unique())
item_num = len(ds['iid'].unique())

test_val = pd.Series(ds['iid'].unique()).sample(frac=0.1, axis=0, random_state=2019)
test_iid = (test_val).sample(frac=0.5)
val_iid  = test_val.drop(labels=test_iid.index)

in_test = ds['iid'].isin(test_iid)
in_val  = ds['iid'].isin(val_iid) 

testset = ds[in_test] 
valset  = ds[in_val]
trainset= ds[~(in_test | in_val)]
trained_user = trainset['uid'].unique()

def clean_testvalset(dataset, trained_user):
    """
        删除掉，dataset中的多个用户，只保留一个
    """
    data = dataset[dataset['uid'].isin(trained_user)]
    return data.groupby('iid').apply(lambda x: x.sample(n=1))

testset = clean_testvalset(testset, trained_user).reset_index(drop=True)
valset  = clean_testvalset(valset , trained_user).reset_index(drop=True)
trainset= trainset.reset_index(drop=True)

#import pdb
#pdb.set_trace()

assert(abs(len(test_iid) - len(val_iid)) <= 1)             # check 个数平均
assert(len(set(test_iid) - set(val_iid)) == len(test_iid)) # check 不相交
assert(testset['uid'].isin(trainset['uid']).all())         # check 都是训练过的user_iid
assert(True)  # TODO trainset和testset的iid不相交
assert(True)  # TODO 验证

testset.to_csv('../data/test.csv')
valset.to_csv('../data/val.csv')
trainset.to_csv('../data/train.csv')


import json
def to_baseline_format(dataset, outfile):
    import pdb
    pdb.set_trace()
    images = []
    dic_img = {}
    set_trainiid = set(trainset['iid'].unique())
    
    for id, uid, iid, review, q in dataset.values.tolist():
        img = dic_img.get(iid, {})
        if iid not in dic_img:
            images.append(img)
        dic_img[iid] = img

        img['iid'] = str(int(iid))
        rs = img.get('reviews', [])
        img['reviews'] = rs
        us = img.get('uids', [])
        img['uids'] = us

        us.append(str(int(uid)))
        rs.append(review)
        if iid in set_trainiid:
            img['split'] = 'train'
        else:
            img['split'] = 'val'
    
    db = {'images' : images}
    assert (db)
    json.dump(db, open(outfile,'w'))

newdb = pd.concat([trainset, testset, valset], axis=0, join='inner')
import pdb
#pdb.set_trace()
to_baseline_format(newdb, '../../baseline/AVA_PCap/AVA_Comments_Full.json')


