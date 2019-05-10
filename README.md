# image_classification_with_pytorch
Overall process and code of image classification with pytorch model

博客中有详细解析: https://blog.csdn.net/weixin_36474809/article/details/90030682

# dataset loder
## load code

    print("Load dataset......")
    image_datasets = {x: customData(img_path='data/',
                                    txt_path=('data/TxtFile/' + x + '.txt'),
                                    data_transforms=data_transforms,
                                    dataset=x) for x in ['train', 'val']}
## label format

 in your_data_folder/Txtfile/ folder

train.txt  与  val.txt


in txt file each line is :
*****121.jpg   (tab （\t）)  0（label）

note that label form 0, not from 1

## simple run for debug

choose 1.jpg and 2.jpg in your_data_folder

in folder your_data/Txtfile train.txt, val.txt


file content:
1.jpg	0
2.jpg	1

modify customData_train.py by your model and your path and dir name
then run:   python customData_train.py

## possible bugs

### python and torch version

/home/xingxiangrui/env/lib/python3.6/site-packages/torchvision/transforms/transforms.py:563: UserWarning: The use of the transforms.RandomSizedCrop transform is deprecated, please use transforms.RandomResizedCrop instead.
"please use transforms.RandomResizedCrop instead.")
/home/xingxiangrui/env/lib/python3.6/site-packages/torchvision/transforms/transforms.py:188: UserWarning: The use of the transforms.Scale transform is deprecated, please use transforms.Resize instead.
"please use transforms.Resize instead.")

solve: change the function name above

### label error
RuntimeError: cuda runtime error (59) : device-side assert triggered at /home/lychee/mycode/pytorch/aten/src/THC/generic/THCTensorMath.cu:24

solve: label from 0, not 1

### index error
return loss.data[0]
IndexError: invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python 

slove: change the code below
initial code:
train_loss+=loss.data[0]
change to:
train_loss+=loss.item()
# dataset and label generate
##  dataset images

    print("Load dataset......")
    image_datasets = {x: customData(img_path='data/',
                                    txt_path=('data/TxtFile/' + x + '.txt'),
                                    data_transforms=data_transforms,
                                    dataset=x) for x in ['train', 'val']}

image stored in your_data/ folder

dataset_and_label_gen.py can help you achieve this

in your_data_dir/TxtFile is train.txt and val.txt

## label format

related code:
class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split('\t')[0]) for line in lines]
            self.img_label = [int(line.strip().split('\t')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

such as:

1.jpg	0
2.jpg	1

## batch process to generate label and copy image to the folder needed


### related code about image copy

source_image_list = os.listdir(source_image_dir)
for idx in range(len(source_image_list)):
    if '.png' in source_image_list[idx-1]:
        continue
    elif '.jpg' in source_image_list[idx-1]:
        continue
    else:
        del source_image_list[idx]

related code about rand dataset for 

 -*- coding: utf-8 -*-
import random
 对list洗牌，在原list上做改变
list = range(10)
print list
random.shuffle(list)
print "随机排序列表 : ",  list

code about split dataset into train and val,
拆为训练集和验证集，分别1/4和3/4

# train list and val list
source_train_list=[]
source_val_list=[]
for idx in range(len(source_image_list)):
    if idx<len(source_image_list)/4:
        source_val_list.append(source_image_list[idx-1])
    else:
        source_train_list.append(source_image_list[idx-1])

图像读出与写入img read and write

图像存于src_img之中，图像重命名用后用save写入。

    # read dource images and rename
    path_source_img = os.path.join(source_image_dir, source_image_name)
    src_img = Image.open(path_source_img)
    full_image_name=prefix+"_train_"+source_image_name
    print(full_image_name)
    # save renamed image to the target dir
    target_image_path=os.path.join(target_image_dir, full_image_name)
    src_img.save(target_image_path)

label generate



# create label_file or write label file
txt_file_train_name="train.txt"
txt_file_val_name="val.txt"
txt_file_train_path=os.path.join(txt_file_dir, txt_file_train_name)
txt_file_val_path=os.path.join(txt_file_dir, txt_file_val_name)
train_txt_file= open(txt_file_train_path,"a")
val_txt_file= open(txt_file_val_path,"a")

有必要对每行加一个"\n"进行结尾

    # write image names and labels
    line_strings= full_image_name+"\t"+str(class_label)+"\n"
    train_txt_file.write(line_strings)

三、训练及验证
3.1 加载数据

直接根据txt文件之中的每一行加载数据，和标签然后可以进行训练。

    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomSizedCrop(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            #transforms.Scale(256),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    use_gpu = torch.cuda.is_available()

    batch_size = 32
    num_class = 3
    print("batch size:",batch_size,"num_classes:",num_class)

    print("Load dataset......")
    # image_datasets = {x: customData(img_path='sin_poly_defect_data/',
    #                                 txt_path=('sin_poly_defect_data/TxtFile/general_train.txt'),
    #                                 data_transforms=data_transforms,
    #                                 dataset=x) for x in ['train', 'total_val']}
    image_datasets={}
    image_datasets['train'] = customData(img_path='sin_poly_defect_data/',
                                         txt_path=('sin_poly_defect_data/TxtFile/general_train.txt'),
                                         data_transforms=data_transforms,
                                         dataset='train')
    image_datasets['val'] = customData(img_path='sin_poly_defect_data/',
                                       txt_path=('sin_poly_defect_data/TxtFile/real_poly_defect.txt'),
                                       data_transforms=data_transforms,
                                       dataset='val')
    # train_data=image_datasets.pop('general_train')
    # image_datasets['train']=train_data
    # val_data=image_datasets.pop('total_val')
    # image_datasets['val']=val_data

    # wrap your data and label into Tensor
    print("wrap data into Tensor......")
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle=True) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("total dataset size:",dataset_sizes)

[点击并拖拽以移动]
3.2 数据加载函数


def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

 define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split('\t')[0]) for line in lines]
            self.img_label = [int(line.strip().split('\t')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label

[点击并拖拽以移动]
3.3 模型训练与验证

定义loss

    print("Define loss function and optimizer......")
    # define cost function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)

    # Decay LR by a factor of 0.2 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.2)

    # multi-GPU
    model_ft = torch.nn.DataParallel(model_ft, device_ids=[0])

[点击并拖拽以移动]
3.4 训练并保存模型

    # train model
    print("start train_model......")
    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=15,
                           use_gpu=use_gpu)

    # save best model
    print("save model......")
    torch.save(model_ft,"output/resnet_on_PV_best_total_val.pkl")
