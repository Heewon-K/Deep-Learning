import os
import argparse
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from histo_clahe import histo_clahe

# ---------------------------------------------------
#                  parser 지정
# ---------------------------------------------------
parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--savedir', default='../image/train',help='save directory')
opt = parser.parse_args()

# ---------------------------------------------------
#                   함수 지정
# ---------------------------------------------------
## cropped_img의 하부 폴더를 image 폴더의 하부폴더로 생성하기 위한 함수
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

## cropped_img 하부 폴더명(=견별라벨)을 읽고 가장 업데이트된 폴더명을 통해 DB에 사용자 입력 정보를 넣기 위한 함수
def new_input():
    global DataBase
    global idx
    dog_id = idx  # 신규 강아지의 일련번호 지정
    
    # ----- 견주가 입력하는 강아지 정보 -----
    # 강아지 이름 - dog_name
    dog_name = input("강아지의 이름 : ")
    
    # 강아지의 나이 - dog_age : int만 입력 
    if len(dog_name) > 0: 
        while True:
            try:
                age_answer = int(input("강아지의 나이 (숫자만 입력하세요): "))
                break
            except ValueError:
                print('올바른 입력값이 아닙니다.')
                continue
        dog_age = age_answer
    
    # 견종 - dog_sp
    dog_sp = input("견종 : ")
    
    # 견주 이름 - owner_name 
    owner_name = input("견주의 이름 : ")
    
    # 견주의 연락처 - owner_info : int만 입력  
    if len(owner_name) > 0:
        while True:
            try:
                info_answer = int(input("견주의 연락처 (숫자만 입력하세요): "))
                break
            except ValueError:
                print('올바른 입력값이 아닙니다.')
                continue
        owner_info = '0' + str(info_answer)  
    
    # 강아지 성별 - dog_sex : 암/암컷 = 0, 수/수컷 = 1              
    if len(owner_info) == 11:
        while True:
            sex_answer = input("강아지의 성별 (암/수) : ")
            if sex_answer not in ['암', '암컷', '수', '수컷']:
                print('올바른 입력값이 아닙니다.')
                continue
            else:
                break
        if sex_answer in ['암', '암컷']:
            dog_sex = 0
        if sex_answer in ['수', '수컷']:
            dog_sex = 1
            
    # 중성화 여부 - dog_neu : n/N = 0, y/Y = 1 
    if dog_sex in [0,1]: # 중성화 여부 입력
        while True:
            neu_answer = input('강아지의 중성화 여부 (Y/N) : ')
            if neu_answer not in ['y', 'Y', 'n', 'N']:
                print('올바른 입력값이 아닙니다.')
                continue
            else:
                break
        if neu_answer in ['n', 'N']:
            dog_neu = 0
        if neu_answer in ['y', 'Y']:
            dog_neu = 1
            
    # 강아지 체중 - dog_wght : float만 입력  
    if dog_neu in [0,1]:
        while True: 
            try:
                wght_answer = float(input('강아지의 체중 (ex. 3, 3.5): '))
                break
            except ValueError:
                print('올바른 입력값이 아닙니다.')
                continue
        dog_wght = wght_answer       
    
    input_data = {'일련번호': dog_id, '강아지이름': dog_name, '나이': dog_age, '견종':dog_sp, '견주이름':owner_name, 
                  '연락처':owner_info, '성별':dog_sex, '중성화':dog_neu, '체중' : dog_wght}
    DataBase.loc[len(DataBase)] = input_data
    DataBase.to_csv('DB.csv', index = False, encoding = 'cp949')
    
# --------------------------------------------------- 
#                  실행 파트   
# ---------------------------------------------------
## cropped_img를 회전 및 크기 조정하여 이미지 데이터 수 늘리기
path = 'image/cropped_img' 
folder_list = os.listdir(path) # cropped_img의 하부 폴더명으로 구성된 리스트 생성
for f in folder_list:
    if 'DS_Store' not in f:    
        file_list = os.listdir(path + '/' + f) # = '../image/cropped_img' + '/' + '폴더명'
        rotate = [0, 15, 30, -15, -30] # 회전값

        for file in file_list: 
            s = os.path.splitext(file) 
            savedir = [] # 원본이미지 및 회전・용량 조정 이미지를 저장하기 위한 디렉터리명(파일명)
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

## 등록 강아지의 정보 DB.csv에 저장
idx = max(folder_list) # = 폴더명 중 가장 큰 것 , 즉 가장 최근에 생성된 폴더명

if not os.path.isfile('DB.csv'):
    DataBase = pd.DataFrame(columns=['일련번호', '강아지이름', '나이', '견종', '견주이름', '연락처', '성별', '중성화', '체중'])
    new_input()

else:
    DataBase = pd.read_csv('DB.csv', encoding = 'cp949')
    new_input()
 
## 등록되었음을 알림
print('등록되었습니다.')