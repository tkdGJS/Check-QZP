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
import snapshot_storing_gzip_n as ss


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
    #                        'occlusion' : int(boxes[9])
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
            'labels': torch.tensor(labels, dtype=torch.int64)  # Convert labels to a tensor
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

#    images_dir = os.listdir(train_sequence_dir)
#    train_images_dir = []
#
#    for path in images_dir:
#        path = os.path.join(train_sequence_dir, path)
#        train_images_dir.append(path)
    #print(os.listdir(train_images_dir[0]))
#    assert False
#    valid_annotations_dir = "./data/DRONE/task1/VisDrone2019-DET-val/annotations/"
#
#    valid_annotations = os.listdir(valid_annotations_dir)
#
#    valid_images_dir = "./data/DRONE/task1/VisDrone2019-DET-val/images/"


    epochs = 10
    BATCH_SIZE = 4
    OUTPUT_PATH = './fio_test/drone-task2-checkpoint/'

    #trainset
    train_list = extraction(train_annotations_dir ,train_annotations ,train_sequence_dir)
#    assert False
    train_dataset = ObjectDetectionDataset(train_list, transform=get_transform_train())

    indices = torch.randperm(len(train_dataset)).tolist()

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    #vaildset
#    valid_list = extraction(valid_annotations_dir ,valid_annotations ,valid_images_dir)
#
#    valid_dataset = ObjectDetectionDataset(valid_list, transform=get_transform_valid())
#
#    indices = torch.randperm(len(valid_dataset)).tolist()
#
#    valid_data_loader = DataLoader(
#        valid_dataset,
#        batch_size=4,
#        shuffle=False,
#        num_workers=0,
#        collate_fn=collate_fn
#        )

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

    backbone = resnet_fpn_backbone('resnet152', True)
    model = FasterRCNN(backbone, num_classes)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4)

    #추가한 코드
    PATH = "checkpoint30.pt"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint_list = []
    epoch_list = []
    model_list=[]



    cost = 0

####################################################################################
#    a = 0
#    for idx, (images, targets) in enumerate(train_data_loader):
#        a = a +len(targets)
#    print(a)


####################################################################################




#manager = Manager()
#activate_snapshot = manager.Value('i', 0)
#images, targets= next(iter(train_data_loader))
#summary(model_resNet50, (3, 1080, 1920))
#print(len(images[0][0][0]))
#    assert False
def main():
    start_model_time = time.time()
    for epoch in range(epochs):
        avg_loss = 0
        start_epoch_time = time.time()
        for idx, (images, targets) in enumerate(train_data_loader):
            #print(idx)
            images = list(image.to(device) for image in images)

            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
#            print(loss_dict)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            avg_loss += loss / len(train_data_loader)

            torch.cuda.empty_cache()

            if idx == 0:
                model_state = model.state_dict()
                op_state = optimizer.state_dict()

                start_checkpoint_time = time.time()
                save = ss.async_checkpointing(model_state, op_state, epoch,OUTPUT_PATH)



                save.snapshot_storing(model_state, op_state, epoch,OUTPUT_PATH)



                end_checkpoint_time = time.time()
                print("check"+str(end_checkpoint_time - script_start))
                checkpoint_delta = np.round(end_checkpoint_time - start_checkpoint_time, 5)
                checkpoint_list.append(checkpoint_delta)

        print('[Epoch: {:>4}) cost = {:>.9}'.format(epoch +1, avg_loss))
        end_epoch_time = time.time()
        epoch_delta = np.round(end_epoch_time - start_epoch_time, 5)
        epoch_list.append(epoch_delta)
    end_model_time = time.time()
    model_delta = np.round(end_model_time - start_model_time)
    model_list.append(model_delta)
#
#
if __name__ == '__main__':
    main()
##########################################################################
#evaluate

#    def plot_image_from_output(image, annotation):
#
#        image = image.permute(1,2,0)
#        
#        fig,ax = plt.subplots(1)
#        
#        for idx in range(len(annotation["boxes"])):
#
#            save_path = f"./objects/output_image_{idx}.jpg"
#            xmin, ymin, xmax, ymax = annotation["boxes"][idx]
#
#            if annotation['labels'][idx] == 1 :
#                rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
#            
#            elif annotation['labels'][idx] == 2 :
#                
#                rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='g',facecolor='none')
#                
#            else :
#            
#                rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='orange',facecolor='none')
#
#            ax.add_patch(rect)
#
#        plt.savefig(save_path)
#
#
#    def evaluate_model(model_resNet50 ,images ,threshold):
#        model_resNet50.eval()  # Set the model to evaluation mode
#        preds = model_resNet50(images)
#        print(preds)
#        for idx, preds in enumerate(preds):
#            idx_list = []
#
##            for i, score in enumerate(preds[idx]['scores']):
##                if score > threshold:
##                    idx_list.append(i)
##
##            preds[idx]['boxes'] = preds[idx]['boxes'][idx_list]
##            preds[idx]['labels'] = preds[idx]['labels'][idx_list]
##            preds[idx]['scores'] = preds[idx]['scores'][idx_list]
#
#        return preds
#
#
#
#
#    with torch.no_grad():
#        for images, targets in valid_data_loader:
#            images = list(image.to(device) for image in images)
#            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#            pred = evaluate_model(model_resNet50, images, 0.1)
#            print(pred) 
#
##            print("Target : ", targets['labels'])
##            plot_image_from_output(imgs[idx], targets[_idx])
##            print("Prediction : ", pred['labels'])
##            for image in images:
##                plot_image_from_output(images, pred)



    print(model_list)
    print(epoch_list)
    print(checkpoint_list)


    torch.save({'model_state_dict' : model.state_dict()}, './base+g.pt')
