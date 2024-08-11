import os, cv2, torch, sys
from torchvision import transforms
from torch.utils import data

def get_data_path(img_dir):
    img_list = []
    label_list = []
    for fn in os.listdir(img_dir):
        img_list.append(os.path.join(img_dir, fn))
        label_list.append(int(fn.split('_')[0])-1)
    return img_list, label_list

def get_data(img_dir):
    # 图像列表和标签
    img_list, label_list = get_data_path(img_dir)
    transform = transforms.Compose([
        #transforms.Resize((112, 112)),
        transforms.ToTensor()  # 第二步转换，作用：第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
    ])

    return Generate_Dataset(img_list, label_list, transform)

def get_data_iter(dataset, batch_size):
    return data.DataLoader(
	dataset,
	batch_size=batch_size,
	shuffle=True
)


class Generate_Dataset(data.Dataset):
	def __init__(self, img_paths, labels, transform):
		self.imgs = img_paths
		self.labels = labels
		self.transforms = transform
		self.classes = len(set(labels))

	# 进行切片
	def __getitem__(self, index):
		img = self.imgs[index]
		label = self.labels[index]
		pil_img = cv2.imread(img)
		pil_img = cv2.resize(pil_img, (112, 112))
		data = self.transforms(pil_img)
		return data, label

	# 返回长度
	def __len__(self):
		return len(self.imgs)