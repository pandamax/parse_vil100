'''
convert to Tusimple json/txt format.
'''

import cv2
import json
import numpy as np
import os

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


def get_mask(mask, label, instance_gap):
    # read label
    label_content = open(label)
   
    label_info = json.load(label_content)['annotations']
    lanes_num = 0
    
    for index, line in enumerate(label_info['lane']):
        lanes_num += 1
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
        points = sorted(points , key=lambda k: (k[1], k[0]))
        
        # print(points)
        while ptEnd < len(points_x):
            mask = cv2.line(mask, points[ptStart], points[ptEnd], [instance_gap * (index+1)]*3, 4, lineType = 8)
            ptStart += 1
            ptEnd +=  1
        
    max_val = lanes_num * instance_gap
            
    return mask, max_val

def lane_instance(label_gray,pix_value, hstart, hend, hdis):
    lane = []
    for hstep in range(hstart, hend, hdis): # 
        # h_samples.append(hstep)
        wids = np.where(label_gray[hstep][:] == pix_value)
        for ele in list(wids):
            # print(list(ele))
            if len(ele) == 0:
                val = -2
            else:
                val = int(sum(ele)/(len(ele))) # get average x_value.
            # if val != 1:
            lane.append(val)
    return lane      

if __name__ == '__main__':
    # choose datasets category from:'train','test'
    datasets_category = 'test'  
    dataset_dir = '/mnt/h/lane_datasets/VIL-100'   
    # datasets dir
    # dataset_dir = '{}/{}/'.format(path_to_datasets, datasets_category)
    # write ground truth in json or txt.
    save_gt = dataset_dir + '/data/{}_converted.json'.format(datasets_category)

    # read file from txt
    txt_file = '{}/data/{}.txt'.format(dataset_dir, datasets_category)

    file_list = open(txt_file)
    for file in file_list:
        file = file.strip()
        full_img_path = dataset_dir + file

        if not os.path.exists(full_img_path):
                continue
        print("Now dealing with:", file)
        file_name = os.path.splitext(file.strip().split('/')[-1])[0] 
        json_file = dataset_dir + file.replace('JPEGImages', 'Json') + '.json'
        
        # if os.path.exists(full_img_path):
        img = cv2.imread(full_img_path)
        
        h = img.shape[0]
        w = img.shape[1]
        
        # set param.
        points_num = 56*3
        instance_gap = 20
        hstart = 0
        hend   = h
        hdis   = h // points_num
        
        img_dict = {}
        h_samples = [] # height
        lanes = []
        
        mask = np.zeros([h,w,3],dtype=np.uint8)
        
        # parse label
        label_mask, max_value = get_mask(mask, json_file,instance_gap)
        
        # convert to grayscale.
        label_gray = label_mask[:,:,1]
        
        for hstep in range(hstart, hend, hdis):
            h_samples.append(hstep)
            
        # neg samples.   
        if  max_value == 0:
            lanes.append([-2]*points_num)
            
        # value:pix_value
        else:
            for value in range(instance_gap, max_value + 1, instance_gap):
                # print("value", value)
                lane = lane_instance(label_gray,value, hstart, hend, hdis)
                
                if max(lane) == -2:
                    lanes.append([-2]*points_num)
                else:
                    lanes.append(lane)

        img_dict["lanes"] = lanes
        img_dict["h_samples"] = h_samples
        img_dict["raw_file"] = f'{file}' # img_path
        
        img_dict_str = str(img_dict)
        # print(img_dict_str)
        img_dict = eval(img_dict_str)
        
        # write to txt
        # with open("save_gt","a+") as f:
        #     f.writelines(img_dict_str + '\n')
        #     f.close()

        # write to json
        with open(save_gt,"a+") as out:
            string = json.dumps(img_dict)
            string += '\n'
            out.write(string)
            out.close()

        # cv2.imencode('.png',label_mask)[1].tofile('{}\{}.png'.format(save_mask_dir,file_name))
        
        
    print("finished~~")



