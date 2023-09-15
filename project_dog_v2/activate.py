import os
import time
import cv2
# 2. crop scan_input
# os.system('python scan/YOLOv5/detect.py --source image/scan_input/0 --weights scan/YOLOv5/best.pt --option scan --save-conf')
# time.sleep(0.5)

# 3. classifier
classifier_result = os.system('python scan/classifier.py --dir image/')
time.sleep(0.5)


# if 분기점



# crop register_input
# if classifier_result.move == "Y" or classifier_result.move =="y":
#     print('a')
# os.system('python register/YOLOv5/detect.py --source image/register_input/borzoi --weights register/YOLOv5/best.pt --option register --save-conf')
# time.sleep(0.5)


# # preprocess.py 실행, DB 추가
# os.system('python register/preprocess.py --savedir image/train')