import cv2, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm1d
from backbones import get_model



def inference(net:nn.Module, img):
    img = cv2.imread(img)
    img = cv2.resize(img, (112, 112))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float().cuda()
    img.div_(255).sub_(0.5).div_(0.5)
    net.cuda()

    return net(img).cpu().detach().numpy()


if __name__ == "__main__":

    test_path = r'./palmvein_ROI/'
    test_list = os.listdir(test_path)

    model = get_model()

    output_dict = {}
    for idx, file in enumerate(test_list):
        output = inference(model, test_path + file)
        output_dict[file] = output[0]

    print('特征提取完成')
        
    congeneric_flag = []
    dist_list = []

    for i in range(len(list(output_dict.keys()))):
        for j in range(i + 1, len(list(output_dict.keys()))):
            output1 = output_dict[list(output_dict.keys())[i]]
            output2 = output_dict[list(output_dict.keys())[j]]

            #   计算二者之间的距离
            l1 = np.linalg.norm(output1 - output2, axis=0) #欧氏距离
            #l1 = cosine_similarity(output1.reshape(1, -1), output2.reshape(1, -1)).reshape(-1) #余弦距离

            dist_list.append(l1)
            congeneric_flag.append(test_list[i].split('-')[0] == test_list[j].split('-')[0])


    # threshold集合，排序算roc
    th_list = np.array(dist_list)
    th_list = np.sort(th_list)
    th_list =np.unique(th_list)

    y_score = np.asarray(dist_list)
    # 协同对
    y_true = np.asarray(congeneric_flag)
    # 非协同对标记
    y_false = (y_true == False)

    with open('{}.txt'.format(test_path.split('/')[1]), 'w', encoding = 'utf-8') as f:
        f.write('th\t\tFAR\t\tFRR' + '\n')
        best_acc = 0
        best_th = 0
        EER = 100
        
        idx = 0
        idx_l = 0
        idx_r = len(th_list)
        idx = (idx_r + idx_l)//2
        start_flag = False
        while True:
            th = th_list[idx]

            # 小于th认为是同一张
            y_test = (y_score < th)
            acc = np.mean((y_test == y_true).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_th = th

            test_acc = (y_test == y_true)
            test_like = (test_acc * y_true)
            test_unlike = (test_acc * y_false)

            FRR = 1. - np.sum(test_like)/np.sum(y_true)
            FAR = 1. - np.sum(test_unlike)/np.sum(y_false)


            if start_flag:
                f.write('%.6f\t\t%.6f\t\t%.6f'%(th, FAR, FRR) + '\n')

                EER = (FRR + FAR)/2

                if FAR == 0.:
                    last_FRR = FRR
                    
                if FRR < FAR:
                    f.write('\n准确率：{}，阈值：{}，等误率：{}，拒真率：{}'.format(best_acc, best_th, EER, last_FRR))
                    break
                
                idx += 1
                
            else:
                if FAR != 0.:
                    idx_r = idx
                else:
                    idx_l = idx
                idx = (idx_r + idx_l)//2
                    
                if idx_l + 1 >= idx_r:
                    start_flag = True
                    idx = idx_l
                    print('拒真率：{}'.format(FRR))
            
                
          
    print('准确率：{}，阈值：{}，等误率：{}，拒真率：{}'.format(best_acc, best_th, EER, last_FRR))  

    
