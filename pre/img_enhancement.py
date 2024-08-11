import os, cv2,  math
import numpy as np
from matplotlib import pyplot as plt

def show_img(img):
    if img.ndim == 2 :
        plt.imshow(img, cmap='gray')
        
    else:
        plt.imshow(img[:,:,2::-1])
    plt.xticks([])
    plt.yticks([])

#3.AGC 伽马自平衡
def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

def AGC(img):
    mean = np.mean(img)
    gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma
    image_gamma_correct = gamma_trans(img, gamma_val)  # gamma变换
    return image_gamma_correct


def img_enhance(img_dir, save_dir, img_name, resize = None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    if len(img.shape) >1:
        img = img[:,:,0]
    img1=AGC(img)

    clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(4,4))
    img2=clahe.apply(img1)

    imgf=np.float32(img2)
    img3=cv2.normalize(imgf,0,1,cv2.NORM_MINMAX)
    img4=cv2.GaussianBlur(img3,ksize=(15,15),sigmaX=4)

    img5=cv2.Laplacian(img4,ddepth=-1,ksize=9)
    img5[img5<0] = 0
    img5[img5>255] = 255
    #归一化：0~255
    #imgf=np.float32(img5)
    img6=cv2.normalize(img5,0,255,cv2.NORM_MINMAX)

    if resize:
        cv2.resize(img, (resize, resize))

    cv2.imwrite(os.path.join(save_dir, img_name.split('.')[0]+'.jpg'), img5*255)
    return img5

if __name__ == '__main__':
    img_dir = r'E:\desktop\Palmvein_ROI_gray_128x128\session1'
    save_dir = r'E:\desktop\Palmvein_ROI_gray_128x128\enhance_img'
    img_name = '1_1.bmp'

    img_list = []
    for fn in os.listdir(img_dir):
        img_list.append(fn)

    for img_name in img_list:
        img_enhance(img_dir, save_dir, img_name)