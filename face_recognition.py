#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys,os,dlib,glob
import numpy as np
from skimage import io
import imutils
import cv2
#人臉圖片路徑
face_data_path = "./resource"
#要辨識的圖片名稱
img_name = sys.argv[ 1]
#載入人臉檢測器
detector = dlib.get_frontal_face_detector()
#人臉68特徵點模型的路徑及檢測器(要先到github下載檔案)
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#人臉辨識模型及檢測器(要先到github下載檔案)
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
#人臉描述list
descriptors = []
#候選人臉名稱list
candidate = []
#讀取人臉face_data_path裡的所有圖片,os.path.join用於拼接檔案路徑
for photo in glob.glob(os.path.join(face_data_path, "*.jpg")):
    base = os.path.basename(photo)
#os.path.splitext()用於分離檔名與副檔名
    candidate.append(os.path.splitext(base)[ 0])
    img = io.imread(photo)
    #人臉偵測
    dets = detector(img, 1)
    
    for i,j in enumerate(dets):
        #68特徵點偵測
        shape = shape_predictor(img, j)
        
        #128特徵向量描述
        face_descriptor = face_rec_model.compute_face_descriptor(img,shape)
        
        #轉換成numpy array格式
        v = np.array(face_descriptor)
        descriptors.append(v)
#對要辨識的目標圖片做相同處理
#讀取照片
img = io.imread(img_name)
#人臉偵測
dets = detector(img, 1)

distance = []

for i,j in enumerate(dets):
        #68特徵點偵測
        shape = shape_predictor(img, j)
        
        #128特徵向量描述
        face_descriptor = face_rec_model.compute_face_descriptor(img,shape)
        
        #轉換成numpy array格式
        d_test = np.array(face_descriptor)
        
        #匡出人臉的四個頂點
        x1 = j.left()
        y1 = j.top()
        x2 = j.right()
        y2 = j.bottom()
        #以方框(紅)匡出人臉
        cv2.rectangle(img, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2. LINE_AA)
        
        #計算歐式距離,線性運算函示
        for i in descriptors:
            dist_ = np.linalg.norm(i - d_test)
            distance.append(dist_)
#zip函數將元素打包並存入dict(候選圖片,距離)
candidate_distance_dict = dict( zip(candidate,distance))
#接著將候選圖片及人名進行排序
candidate_distance_dict_sorted = sorted(candidate_distance_dict.items(), key = lambda d:d[ 1])
# 最短距離為辨識出的對象
result = candidate_distance_dict_sorted[ 0][ 0]
# 在方匡旁標上人名
cv2.putText(img, result, (x1, y1), cv2. FONT_HERSHEY_SIMPLEX , 1, ( 255, 255, 255), 2, cv2. LINE_AA)

img = imutils.resize(img, width = 500)
img = cv2.cvtColor(img,cv2. COLOR_BGR2RGB)
cv2.imshow( "outcome", img)

print(candidate_distance_dict_sorted)
print(result)

cv2.waitKey( 0)
cv2.destroyAllWindows()        

