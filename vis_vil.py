'''
visualize VIL-100 datasets in points form or curves form.

datasets name:vil-100 
paper link: https://arxiv.org/abs/2108.08482
reference: https://github.com/yujun0-0/MMA-Net/tree/main/dataset

datasets structure:
VIL-100
    |----Annotations
    |----data
    |----JPEGImages
    |----Json
    |----train.json
   
*********** A sample of one json-file ***********
{
       "camera_id": 8272,
       "info": {
           "height": 1080 , 
           "width": 1920,
           "date": "2020-11-24",
           "image_path": "0_Road014_Trim005_frames/XXXXXX.jpg"
       },
       "annotations": {
           "lane": [{
   	        "id": 1, 
   	        "lane_id": 1,
   	        "attribute": 1,
   	        "occlusion": 0,
   	        "points": [[412.6, 720],[423.7, 709.9], ...]
           }, {...}, {...}, {...}]
       }
  }
'''

import os 
import cv2
import numpy as np
import json

color = [(218,112,214), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255),
     (100, 255, 0), (100, 0, 255), (255, 100, 0), (0, 100, 255), (255, 0, 100), (0, 255, 100)]

def get_points(mask, label):
    # read label
    label_content = open(label)
   
    label_info = json.load(label_content)['annotations']
    
    # label_info = eval(label_info)
    for index, line in enumerate(label_info['lane']):
        # print(line)
        points_x = []
        points_y = []
        # get points
        for point in line['points']:
            points_x.append(int(float(point[0])))
            points_y.append(int(float(point[1])))
        
        ptStart = 0
    
        points = list(zip(points_x, points_y))
        # sort along y
        sorted(points , key=lambda k: (k[1], k[0]))
        
        # print(points)
        while ptStart < len(points_x):
            image = cv2.circle(mask, points[ptStart], 5, color[index], -1)
            ptStart += 1
            
    return image
 
 
 
def get_curves(mask, label):
    # read label
    label_content = open(label)
   
    label_info = json.load(label_content)['annotations']
    
    # label_info = eval(label_info)
    for index, line in enumerate(label_info['lane']):
        # print(line)
        points_x = []
        points_y = []
        # get points
        for point in line['points']:
            points_x.append(int(float(point[0])))
            points_y.append(int(float(point[1])))
        
        ptStart = 0
        ptEnd =  1
        
        points = list(zip(points_x, points_y))
        # sort along y
        sorted(points , key=lambda k: (k[1], k[0]))
        
        # print(points)
        while ptEnd < len(points_x):
            mask = cv2.line(mask, points[ptStart], points[ptEnd], color[index], 4, lineType = 8)
            ptStart += 1
            ptEnd +=  1
            
    return mask 

if __name__ == '__main__':   
    # choose vis_mode between 'points' and 'curves'
    vis_mod = 'curves'  
    # datasets dir
    dataset_dir = '/mnt/h/lane_datasets/VIL100'
    # save label dir(mask)
    save_mask_dir = '{}/{}_{}'.format(dataset_dir, "vis_datasets", vis_mod)
    if not os.path.exists(save_mask_dir):
        os.makedirs(save_mask_dir)
        
    # read file from txt
    txt_file = dataset_dir  + '/data/train.txt'
    file_list = open(txt_file)
    for file in file_list:
        file = file.strip()
        full_img_path = dataset_dir + file
        
        if not os.path.exists(full_img_path):
            continue
        print("Now dealing with:", file)
        file_name = os.path.splitext(file.strip().split('/')[-1])[0] # image_name xxx
        json_file = dataset_dir + file.replace('JPEGImages', 'Json') + '.json'
        
        img = cv2.imread(full_img_path)
        
        # datasets have different height and width.
        # get img shape,h and w. 
        h = img.shape[0]
        w = img.shape[1]
        
        # parse label
        # visulize points
        if vis_mod == 'points':
            label_mask = get_points(img, json_file)
        else:
            # visulize curves
            label_mask = get_curves(img, json_file)
        
        cv2.imencode('.png',label_mask)[1].tofile('{}/{}.png'.format(save_mask_dir,file_name))
        
    print("finished~~")

    
    
    
    
    
    
    
    



