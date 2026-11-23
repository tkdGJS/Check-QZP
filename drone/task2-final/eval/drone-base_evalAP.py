import torch
import os
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock
import Uniform as U
from torch.utils.data import random_split, DataLoader, Dataset
import time
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as tt
import torchvision.models as models
from torch.utils.data import random_split, DataLoader, Dataset
import timm
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#from torchvision.models.detection import fasterrcnn_resnet50_fpn
#from torchvision.models import ResNet152_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
import snapshot_storing_n as ss
import sklearn.metrics
from collections import Counter
try:
    set_start_method('spawn')
except RuntimeError:
    pass
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


classes = {0:'ignored regions',1:'pedestrian',2:'people',3:'bicycle',4:'car',5:'van',6:'truck',7:'tricycle',8:'awning-tricycle',9:'bus',10:'motor',11:'others'}





def extraction(annotations_dir, annotations, sequence_dir):#sequence_dir is image folders 
    data=[]
    for annotation in annotations:
        with open(os.path.join(annotations_dir ,annotation), 'r') as file:
            lines = file.readlines()
            file_name = annotation.rsplit('.txt',1)[0]
            for line in lines:
#                if len(lines)<15:
                boxes = line.strip().split(',')
                if int(boxes[4]) != 0 and int(boxes[5]) != 0:

#                        print(os.path.join(sequence_dir,file_name, str(boxes[0]).zfill(7) + '.jpg'))
                        data.append({
                            'sequence_id' : file_name,
                            'image_id' : str(boxes[0]).zfill(7),
                            'image_path': os.path.join(sequence_dir,file_name ,str(boxes[0]).zfill(7) + '.jpg'),
                            'frame_index' : int(boxes[0]),
                            'target_id' : int(boxes[1]),
                            'x_min' : int(boxes[2]),
                            'y_max' : int(boxes[3]),
                            'x_max' : int(boxes[2])+int(boxes[4]),
                            'y_min' : int(boxes[3])+int(boxes[5]),
                            'score' : int(boxes[6]),
                            'class_label' : int(boxes[7])
    #                        'truncation' : int(boxes[8]),
#                            'occlusion' : int(boxes[9])
                            })

    return data

def unique(data_list):
    unique_data_dict = {}
    unique_data_list = []

    for record in data_list:
        image_id = record['image_id']
        if image_id not in unique_data_dict:
            unique_data_dict[image_id] = record
            unique_data_list.append(record)

    return unique_data_list

class ObjectDetectionDataset(Dataset):
    def __init__(self, data_list, transform=None):

        self.unique_list = unique(data_list)
        self.data_list = data_list
        self.transform = transform
        self.idx = range(len(unique(data_list)))
        #print(unique(data_list))


    def __len__(self):
        return len(self.unique_list)

    def __getitem__(self, idx):
        data_list = self.data_list
        sample = self.unique_list[idx]
        image_id = sample['image_id']

        image_path = sample['image_path']

#        iscrowd = sample['occlusion']

        image = cv2.imread(image_path)


        matching_samples = [s for s in data_list if s['image_id'] == image_id and s['image_path'] == image_path]

        boxes = []  # List to store bounding boxes
        labels = []  # List to store class labels

        for match in matching_samples:
            x_min = match['x_min']
            y_min = match['y_min']
            x_max = match['x_max']
            y_max = match['y_max']
            class_label = match['class_label']
#            target = match['target_id']

            boxes.append([x_min, y_max, x_max, y_min])
            labels.append(class_label)

        

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),  # Convert boxes to a tensor
            'labels': torch.tensor(labels, dtype=torch.int64),
        }

#        print(image_path,"\n")
#        print(boxes,"\n")

        if self.transform:
            sample = {
                    'image' : image.astype(np.float32),
                    'bboxes' : target['boxes'],
                    'labels' : target['labels']
                    }

            sample = self.transform(**sample)
            image = sample['image']
            image = image.expand(3, -1, -1)

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1,0)

        return image, target


def get_transform_train():
    return A.Compose([
#        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(p=1.0)
    ], bbox_params={'format':'pascal_voc', 'label_fields': ['labels']})

def get_transform_valid():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields':['labels']})


def collate_fn(batch):
    return tuple(zip(*batch))

script_start = time.time()
print(os.getpid())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':

    train_annotations_dir = "./data/DRONE/task2/VisDrone2019-VID-train/annotations/"

    train_annotations = os.listdir(train_annotations_dir)

    train_sequence_dir = "./data/DRONE/task2/VisDrone2019-VID-train/sequences/"

    valid_annotations_dir = "./data/DRONE/task2/VisDrone2019-VID-val/annotations/"

    valid_annotations = os.listdir(valid_annotations_dir)

    valid_sequence_dir = "./data/DRONE/task2/VisDrone2019-VID-val/sequences/"


    epochs = 0
    BATCH_SIZE = 4
    OUTPUT_PATH = './fio_test/drone-task2-checkpoint/'

    #trainset
    train_list = extraction(train_annotations_dir ,train_annotations ,train_sequence_dir)
#    assert False
    train_dataset = ObjectDetectionDataset(train_list, transform=get_transform_train())

    indices = torch.randperm(len(train_dataset)).tolist()

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    valid_list = extraction(valid_annotations_dir ,valid_annotations ,valid_sequence_dir)
#    assert False
    valid_dataset = ObjectDetectionDataset(valid_list, transform=get_transform_train())

    indices = torch.randperm(len(valid_dataset)).tolist()

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )


    #박스 매핑 확인
##############################################################3
#    images, targets= next(iter(train_data_loader))
#    #print(targets)
#    #assert False
#
#    images = list(image.to(device) for image in images)
#    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#    #print(images)
#    plt.figure(figsize=(20,20))
#    for i, (image, target) in enumerate(zip(images, targets)):
#        plt.subplot(2,2, i+1)
#        boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
#        sample = images[i].permute(1,2,0).cpu().numpy()
#        sample = sample*225
#        names = targets[i]['labels'].cpu().numpy().astype(np.int64)
#    #    print(i)
#
#        output_path = f"./objects/output_image_{i}.jpg"
#
#        #원본 이미지
#        #cv2.imwrite(output_path, sample)
#
#    #    print(sample)
#
#        names = targets[i]['labels'].cpu().numpy().astype(np.int64)
#        for i,box in enumerate(boxes):
#            cv2.rectangle(sample,
#                          (box[0], box[1]),
#                          (box[2],box[3]),
#                          (0, 0, 220), 2)
#            cv2.putText(sample, classes[names[i]], (box[0],box[1]+15),cv2.FONT_HERSHEY_COMPLEX ,0.5,(0,220,0),1,cv2.LINE_AA)
#
#        plt.axis('off')
#
#        #mapping 이미지
#        cv2.imwrite(output_path, sample)
#
#   ####################################################################################
#
#
    #assert False

    num_classes = len(classes)

    backbone = resnet_fpn_backbone('resnet152', pretrained=True)
    model = FasterRCNN(backbone, num_classes)




    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4)


    #추가한 코드
    PATH = "base+q+p.pt"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
#    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    checkpoint_list = []
    epoch_list = []
    model_list=[]



    cost = 0
if __name__ == '__main__':
##########################################################################



    print(model_list)
    print(epoch_list)
    print(checkpoint_list)


    def make_prediction(model, img, threshold):
        model.eval()
        preds = model(img)
        for id in range(len(preds)) :
            idx_list = []

            for idx, score in enumerate(preds[id]['scores']) :
                if score > threshold : 
                    idx_list.append(idx)

            preds[id]['boxes'] = preds[id]['boxes'][idx_list]
            preds[id]['labels'] = preds[id]['labels'][idx_list]
            preds[id]['scores'] = preds[id]['scores'][idx_list]

        return preds


#    with torch.no_grad(): 
#        # 테스트셋 배치사이즈= 2
#        for imgs, annotations in valid_data_loader:
#            imgs = list(img.to(device) for img in imgs)
#            print(annotations, "\n")
#            print('and\n')
#            pred = make_prediction(model, imgs, 0.7)
#
#            print(tuple(pred), "\n")

#--------------------------------------------------------------------------------------
#Image Draw
#
#            images = imgs
#            targets = annotations
#            preds = tuple(pred)
#            
#            plt.figure(figsize=(20,20))
#            for i, (image, target, pred) in enumerate(zip(images, targets, preds)):
#                plt.subplot(2,2, i+1)
#                boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
#                sample = images[i].permute(1,2,0).cpu().numpy()
#                sample = sample*225
#                names = targets[i]['labels'].cpu().numpy().astype(np.int64)
#            #    print(i)
#
#                boxes_p = preds[i]['boxes'].cpu().numpy().astype(np.int32)
#                sample_p = images[i].permute(1,2,0).cpu().numpy()
#                names_p = preds[i]['labels'].cpu().numpy().astype(np.int64)
#
#                output_path = f"./objects/output_image_{i}.jpg"
#
#                #원본 이미지
#                #cv2.imwrite(output_path, sample)
#
#            #    print(sample)
#
#                names = targets[i]['labels'].cpu().numpy().astype(np.int64)
#                for i,(box,box_p) in enumerate(zip(boxes, boxes_p)):
#                    cv2.rectangle(sample,
#                                  (box[0], box[1]),
#                                  (box[2],box[3]),
#                                  (0, 0, 220), 2)
#                    cv2.putText(sample, classes[names[i]], (box[0],box[1]+15),cv2.FONT_HERSHEY_COMPLEX ,0.5,(0,220,0),1,cv2.LINE_AA)
#
#                    cv2.rectangle(sample,
#                                  (box_p[0], box_p[1]),
#                                  (box_p[2],box_p[3]),
#                                  (0, 220, 0), 2)
#                    cv2.putText(sample, classes[names[i]], (box_p[0],box_p[1]+15),cv2.FONT_HERSHEY_COMPLEX ,0.5,(0,220,0),1,cv2.LINE_AA)
#                plt.axis('off')
#
#                #mapping 이미지
#                cv2.imwrite(output_path, sample)
#            break
#--------------------------------------------------------------------------------------
    def getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    def getUnionAreas(boxA, boxB, interArea=None):
        area_A = getArea(boxA)
        area_B = getArea(boxB)
        
        if interArea is None:
            interArea = getIntersectionArea(boxA, boxB)
            
        return float(area_A + area_B - interArea)

    def getIntersectionArea(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    def boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    def iou(boxA, boxB):
        # if boxes dont intersect
        if boxesIntersect(boxA, boxB) is False:
            return 0
        interArea = getIntersectionArea(boxA, boxB)
        union = getUnionAreas(boxA, boxB, interArea=interArea)
        
        # intersection over union
        result = interArea / union
        assert result >= 0
        return result


    def mAP(result):
        if not result:
            return 0

        ap = 0
        for r in result:
            ap += r['AP']
        m_AP = ap / len(result)
#        m_AP = np.mean(result)
       
        return m_AP



    def AP(label_class, label_boxes, pred_class, pred_boxes, Class, IOUThreshold = 0.3, method = 'AP'):
        
        # 클래스별 AP, Precision, Recall 등 관련 정보를 저장할 리스트
        result = []
        
        # 클래스별로 접근
        for c in Class:

#            print('c =', c,'\n')
#            print('pc = ', pred_class, '\n')
#            print('lc = ', label_class, '\n')
            
            # 각 class에 따라 박스 분류 => class 별 AP 계산하기 위함
            dects = [pred_boxes[idx] for idx, p in enumerate(pred_class) if p == c]
            gts = [label_boxes[idx] for idx, l in enumerate(label_class) if l == c]
#            print('dects = ',dects,'\n')
#
#            print('gts = ', gts,'\n')
            npos = len(gts)

            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
            

            # 전체 detected box
            for d in range(len(dects)):


                gt = gts
                iouMax = 0

                # 가장 큰 IoU 값을 가지는 박스에 대해 매칭
                for j in range(len(gt)):
                    iou1 = iou(dects[d], gt[j])
                    if iou1 > iouMax:
                        iouMax = iou1
                        jmax = j

                # IoU 및 class 예측 성공  => TP
                # else  => FP
                if iouMax >= IOUThreshold:
                    if label_class[jmax] == pred_class[d]:
                        TP[d] = 1
##                        det[dects[d]][jmax] = 1
                    else:
                        FP[d] = 1
                else:
                    FP[d] = 1

            # Precision, Recall 값을 구하기 위한 누적 TP, FP
            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)

#            print('npos = ', npos)

            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))

            if method == "AP":
                [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)
            else:
                [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)

            r = {
                'class' : c,
                'precision' : prec,
                'recall' : rec,
                'AP' : ap,
                'interpolated precision' : mpre,
                'interpolated recall' : mrec,
                'total positives' : npos,
                'total TP' : np.sum(TP),
                'total FP' : np.sum(FP)
            }
#            t_TP = np.sum(TP)
#            t_FP = np.sum(FP)
#            print('t_TP = ', t_TP,'\n')
#            print('t_FP = ', t_FP,'\n')
#            r = t_TP/t_TP+t_FP


            result.append(r)

        return result
#        """
#        return 0

    def calculateAveragePrecision(rec, prec):
        
        mrec = [0] + [e for e in rec] + [1]
        mpre = [0] + [e for e in prec] + [0]

        for i in range(len(mpre)-1, 0, -1):
            mpre[i-1] = max(mpre[i-1], mpre[i])

        ii = []

        for i in range(len(mrec)-1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i+1)

        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i-1]) * mpre[i])
        
        return [ap, mpre[0:len(mpre)-1], mrec[0:len(mpre)-1], ii]

    def ElevenPointInterpolatedAP(rec, prec):
        mrec = [e for e in rec]
        mpre = [e for e in prec]

        # recallValues = [1.0, 0.9, ..., 0.0]
        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp, recallValid = [], []


        for r in recallValues:
            # r : recall값의 구간
            # argGreaterRecalls : r보다 큰 값의 index
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            #print(r, argGreaterRecalls)

            # precision 값 중에서 r 구간의 recall 값에 해당하는 최댓값
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])

            recallValid.append(r)
            rhoInterp.append(pmax)

        ap = sum(rhoInterp) / 11
        
        return [ap, rhoInterp, recallValues, None]


    
    Class = ['0','1','2','3','4','5','6','7','8','9','10','11']
    m_AP = []
    with torch.no_grad(): 
        # 테스트셋 배치사이즈= 2
        for imgs, annotations in valid_data_loader:
            imgs = list(img.to(device) for img in imgs)
            label_boxes = list(annotations[0].values())[0]

            label_boxes = label_boxes.tolist()

            label_class = list(annotations[0].values())[-1]

            label_class = label_class.tolist()
            
            label_class = list(map(str, label_class))

            #eval    
            pred = make_prediction(model, imgs, 0.5)

            pred_boxes = list(tuple(pred[0].values()))[0]

            pred_boxes = pred_boxes.tolist()

            pred_class = list(tuple(pred[0].values()))[-2]

            pred_class = pred_class.tolist()

            pred_class = list(map(str, pred_class))

 #           print(pred_boxes[0])
            
            m__AP = mAP(AP(label_class, label_boxes, pred_class, pred_boxes, Class, IOUThreshold = 0.3, method = '11'))
#            print(mAP(AP(label_class, label_boxes, pred_class, pred_boxes, Class, IOUThreshold = 0.3, method = 'AP')), '\n')
            m_AP.append(mAP(AP(label_class, label_boxes, pred_class, pred_boxes, Class, IOUThreshold = 0.3, method = '11')))
            print(m__AP,'\n')
#            assert m__AP < 1 or nath.isnan(m__AP)

#            break
#--------------------------------------------------

    print(np.mean(m_AP))
