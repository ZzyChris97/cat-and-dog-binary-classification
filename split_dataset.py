import os
import random
import shutil


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


if __name__ == '__main__':

    random.seed(1)

    dataset_dir = "data"
    split_dir = os.path.join("data", "dogcat_split")
    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    # 方法是一个简单易用的文件、目录遍历器
    '''
    os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。
    root 所指的是当前正在遍历的这个文件夹的本身的地址
    dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    '''
    for root, dirs, files in os.walk(dataset_dir):


        # listdir()返回指定的文件夹包含的文件或文件夹的名字的列表
        # imgs = os.listdir(os.path.join(root, sub_dir))
        # filter() 函数用于过滤序列，过滤掉不符合条件的元素
        imgs_cat = list(filter(lambda x: x.endswith('.jpg') and x.startswith('cat'), files))
        imgs_dog = list(filter(lambda x: x.endswith('.jpg') and x.startswith('dog'), files))
        random.shuffle(imgs_cat)
        random.shuffle(imgs_dog)
        img_cat_count = len(imgs_cat)
        img_dog_count = len(imgs_dog)

        train_cat_point = int(img_cat_count * train_pct)
        valid_cat_point = int(img_cat_count * (train_pct + valid_pct))
        train_dog_point = int(img_dog_count * train_pct)
        valid_dog_point = int(img_dog_count * (train_pct + valid_pct))

        for i in range(img_cat_count):
            if i < train_cat_point:
                out_dir = os.path.join(train_dir, 'cat')
            elif i < valid_cat_point:
                out_dir = os.path.join(valid_dir, 'cat')
            else:
                out_dir = os.path.join(test_dir, 'cat')

            makedir(out_dir)

            target_path = os.path.join(out_dir, imgs_cat[i])
            src_path = os.path.join(dataset_dir, imgs_cat[i])

            shutil.copy(src_path, target_path)

        print('Class:{}, train:{}, valid:{}, test:{}'.format('cat', train_cat_point, valid_cat_point-train_cat_point,
                                                                 img_cat_count-valid_cat_point))
        for i in range(img_dog_count):
            if i < train_dog_point:
                out_dir = os.path.join(train_dir, 'dog')
            elif i < valid_dog_point:
                out_dir = os.path.join(valid_dir, 'dog')
            else:
                out_dir = os.path.join(test_dir, 'dog')

            makedir(out_dir)

            target_path = os.path.join(out_dir, imgs_dog[i])
            src_path = os.path.join(dataset_dir, imgs_dog[i])

            shutil.copy(src_path, target_path)

        print('Class:{}, train:{}, valid:{}, test:{}'.format('dog', train_dog_point, valid_dog_point-train_dog_point,
                                                                 img_dog_count-valid_dog_point))