# to use the torchvision.datasets.ImageFolder函数，所以使用这个工具将ava中的style_list转化为正常的代码
import os
import shutil
dirpath = '../data/ava_dataset/'
imagepath = dirpath + 'images/'
style_dir_path = dirpath + 'style_image_lists/'
train_id = style_dir_path + 'train.jpgl'
train_tag = style_dir_path + 'train.lab'
test_id = style_dir_path + 'test.jpgl'
test_tags = style_dir_path + 'test.multilab'


out_dir = '../data/ava_style/'
if not os.path.exists(out_dir): 
    os.mkdir(out_dir)
out_train_dir = out_dir + 'train/'
out_test_dir = out_dir + 'test/'

if not os.path.exists(out_dir+'train/'): 
    os.mkdir(out_dir+'train/')
if not os.path.exists(out_dir+'test/'): 
    os.mkdir(out_dir+'test/')

fp_id_name = open(style_dir_path + 'styles.txt')
tag2name = dict()
for line in fp_id_name.readlines():
    fields = line.strip().split(' ')
    tag2name[fields[0]] = fields[1] # str 2 str
    if not os.path.exists(out_dir+'train/'+fields[1]): 
        os.mkdir(out_dir+'train/' + fields[1])
    if not os.path.exists(out_dir+'test/'+fields[1]): 
        os.mkdir(out_dir+'test/'  + fields[1])
        

# copy the images to the corresponding directory : train set
trainids = open(train_id).readlines()
traintags = open(train_tag).readlines()
for iid, tag in zip(trainids, traintags):
    iid, tag = iid.strip(), tag.strip()
    tagname = tag2name[tag]
    shutil.copyfile(imagepath + iid + '.jpg', out_train_dir + tagname + '/' + iid + '.jpg')
