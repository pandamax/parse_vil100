'''
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

def get_mask(mask, label, instane_gap, thickness):
    # read label
    label_content = open(label)
   
    label_info = json.load(label_content)['annotations']
    
    for index, line in enumerate(label_info['lane']):
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
            mask = cv2.line(mask, points[ptStart], points[ptEnd], [(index+1)*instane_gap]*3, thickness, lineType = 8)
            ptStart += 1
            ptEnd +=  1
            
    return mask
        
if __name__ == '__main__':
  # choose datasets category from:'train','test'
  datasets_category = 'train'  
  # datasets dir
  dataset_dir = '/mnt/h/lane_datasets/VIL100'

  # save label dir(mask)
  save_mask_dir = dataset_dir + '/mask'
  if not os.path.exists(save_mask_dir):
      os.makedirs(save_mask_dir)
      
      
  # read file from txt
  txt_file = '{}/data/{}.txt'.format(dataset_dir, datasets_category)
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
      
      # get img shape,h and w. 
      h = img.shape[0]
      w = img.shape[1]
      
      # set params
      instane_gap = 30
      thickness = w // 128 
      
      mask = np.zeros([h,w,3],dtype=np.uint8)
      # parse label
      label_mask = get_mask(mask, json_file, instane_gap, thickness)
      
      cv2.imencode('.png',label_mask)[1].tofile('{}/{}.png'.format(save_mask_dir,file_name))
      
      
  print("Done!")

    
    
    
    
    
    
    
    



