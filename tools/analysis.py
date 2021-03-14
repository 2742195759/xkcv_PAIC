import pandas as pd
dataset_path = '../data/dataset.txt'

filt = True

ds = pd.read_csv(dataset_path, index_col=0)
print (ds[0:10])


if filt == True:
    print ('============================筛选用户 + 筛选图 =======================')
    user_len = ds.groupby(by=['uid']).apply(len)
    user_set = set(user_len[user_len>=30].index)
    """ 
        1.make sure only the user is selected, delete the less activate user
    """
    # print (ds['review'])

    ds = ds[ds['uid'].isin(user_set)]
    ds.reset_index(drop=True)
    ds.to_csv('../data/dataset_final.csv')
    print ('==========================筛选结束===========================')

# start process 
user_num = len(ds['uid'].unique())
item_num = len(ds['iid'].unique())
interact_num = len(ds)
interact_rate = interact_num * 1.0 / (user_num * item_num)

print ('============================基本信息===============================')

user_len = ds.groupby(by=['uid']).apply(len)
user_set = set(user_len[user_len>=5].index)
print ('user_len_datase', user_len)
print ('user_num', 'item_num', 'interact_num', 'interact_rate')
print (user_num, item_num, interact_num, interact_rate)
assert (len(user_len[user_len>=1]) == user_num)
assert (len(user_set) == len(user_len[user_len>=5]))

print ('>5 users', len(user_len[user_len>=5]))
print ('>10 users', len(user_len[user_len>=10]))
print ('>20 users', len(user_len[user_len>=20]))

print ('============================同图不同User Caption=====================')
image_len = ds.groupby(by=['iid']).apply(len).sort_values(ascending=False)
print (image_len[0:10])
idx = 0 # 前idx个图片的所有评论
(ds[ds['iid'] == image_len.index[idx]]).to_csv('../data/case_imageid.csv')
print ()
print ('Concrete Reviews in ../data/case_imageid.csv')


print ('=============================验证，没有一个对同一个图片两个评论========')
tmp = ds.groupby(by=['uid','iid']).apply(len)
print ('这么多个', len(tmp[tmp>1]), '是重复评论')
print ('示例:')
print (ds[ds.duplicated(subset=['uid','iid'], keep=False)][0:10].sort_values(by=['uid','iid']).reset_index(drop=True))
