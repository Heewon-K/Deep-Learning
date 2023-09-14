import os
import argparse
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from histo_clahe import histo_clahe

# ---------------------------------------------------
parser = argparse.ArgumentParser(description='Argparse Tutorial')
# parser.add_argument('--dir', default='0',help='dataset directory') -> reference에 존재, 본 script 불용
parser.add_argument('--savedir', default='../image/train',help='save directory')
opt = parser.parse_args()

# ---------------------------------------------------
# cropped_img의 하부 폴더를 image 폴더의 하부폴더로 생성하기 위한 함수
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# cropped_img 하부 폴더명(=견별라벨)을 읽고 가장 업데이트된 폴더명을 통해 DB에 사용자 입력 정보를 넣기 위한 함수
# DB update
def new_input():
    global DataBase
    global idx
    # 신규 강아지 input
    dog_name = input("강아지의 이름 : ")
    dog_age = input("강아지의 나이 : ")
    dog_sp = input("견종 : ")
    owner_name = input("견주의 이름 : ")
    owner_info = input("견주의 연락처 : ")
    dog_med1 = input("강아지의 의료정보 1 : ")
    dog_med2 = input("강아지의 의료정보 2 : ")
    dog_id = idx
    
    input_data = {'일련번호': dog_id, '강아지이름': dog_name, '나이': dog_age, '견종':dog_sp, '견주이름':owner_name, 
                  '연락처':owner_info, '의료정보1':dog_med1, '의료정보2':dog_med2}
    DataBase = DataBase.append(input_data, ignore_index = True)
    DataBase.to_csv('../DB.csv', header = True, index = False)
    
# ---------------------------------------------------

path = '../image/cropped_img' 
folder_list = os.listdir(path) #cropped_img의 하부 폴더명으로 구성된 리스트 생성
for f in folder_list:
    if 'DS_Store' not in f:    
        file_list = os.listdir(path + '/' + f) # = '../image/cropped_img' + '/' + '폴더명'
        rotate = [0, 15, 30, -15, -30] #회전값

        for file in file_list: 
            s = os.path.splitext(file) 
            savedir = [] #원본이미지 및 회전・용량 조정 이미지를 저장하기 위한 디렉터리명(파일명)
            createFolder(opt.savedir + '/' + f) # = '../image/train' + '/' + '폴더명'

            for i in range(10): 
                savedir.append(opt.savedir + '/' + f + '/' + file[:-4]+ '-' +str(i) + '.jpg' )
                # = '../image/train'  + '/' + 폴더명 + '/' + (파일명 - .jpg) + '-' + str(i) + '.jpg'
 
            #사이즈 1 rotate 저장
            img = histo_clahe(path + '/' + f + '/' + file) #이미지 입력
            height, width, channel = img.shape

            for i in range(5):
                matrix = cv2.getRotationMatrix2D((width/2, height/2), rotate[i], 1)
                dst = cv2.warpAffine(img, matrix, (width, height))
                cv2.imwrite(savedir[i],dst)

            #사이즈 1/2 rotate 저장
            img = cv2.resize(img,(int(width / 2), int(height / 2)))
            height, width, channel = img.shape

            for i in range(5):
                matrix = cv2.getRotationMatrix2D((width/2, height/2), rotate[i], 1)
                dst = cv2.warpAffine(img, matrix, (width, height))
                cv2.imwrite(savedir[i+5],dst)

idx = max(folder_list) # = 폴더명 중 가장 큰 것 
print(idx)
if not os.path.isfile('../DB.csv'):
    DataBase = pd.DataFrame(columns=['일련번호', '강아지이름', '나이', '견종', '견주이름', '연락처', '의료정보1', '의료정보2'])
    new_input()
else:
    DataBase = pd.read_csv('../DB.csv')
    new_input() 
# ------------------------- 바뀐 점----------------
print('등록되었습니다!')