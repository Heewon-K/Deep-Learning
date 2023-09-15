import os
import numpy as np
import cv2
import pandas as pd

import pickle
import sklearn
import argparse
import time

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from histo_clahe import histo_clahe
from os import listdir
from os.path import isfile, join





# parsing(=구문분석) : 스크립트 내용을 토큰화하여 분석
# parsing은 compile과도 밀접한 관련, 기계어에 대한 번역에 선행
# argparser : python 실행 시 서로 다른 옵션을 상황에 따라 실행할 때 사용

# Argument parser 초기화 = argparser 객체 선언
parser = argparse.ArgumentParser(description='Argparse Tutorial')
# ArgumentParser 클래스의 객체를 생성합니다. 
# 이 객체는 명령행 인수를 정의하고 파싱하는 데 사용됩니다.
# description 매개변수를 사용하여 스크립트에 대한 간단한 설명을 추가할 수 있습니다.

parser.add_argument('--dir', default='image', # 경로 변경 필요
                     help='dataset directory')
# add_argument 메서드를 사용하여 명령행 인수를 정의합니다. = argument 추가 
# 이 예제에서는 --dir 인수를 정의하고, 기본값으로 'image'를 설정하고, 인수에 대한 설명을 제공합니다.
# 즉, 사용자가 --dir로 값을 지정하지 않으면 기본값인 'image'가 사용됩니다.
# --dir 인자일 때 dataset directory


# parser.add_argument('--test', default='test_1.jpg' # 경로 변경 필요
                    # , help='test image data')
#  비슷하게 --test 인수를 정의하고, 기본값으로 'test_1.jpg'를 설정하고,
#  인수에 대한 설명을 제공합니다.
# --test 인자일 때 test image data

opt = parser.parse_args()
# parse_args 메서드를 호출하여 명령행에서 전달된 인수를 파싱하고, 이를 opt 변수에 저장합니다.
# 이제 opt 객체는 파싱된 인수를 포함하고 있습니다.

# 학습 데이터를 읽어 이미지 리스트 X와 라벨리스트 Y를 반환하는 함수
# 훈련 시에만 활용
def read_data(label2id):
    X = [] #이미지 라벨 저장
    Y = [] #이미지 라벨 저장
    for label in os.listdir('image/train'):#/Users/mo3n/Documents/project/dog/git2/project_dog/
        # image 디렉터리 내의 각 라벨(폴더)에 대해 반복합니다.#/Users/mo3n/Documents/project/dog/git2/project_dog/
        # 라벨(폴더)명은 int(현재 0~5)
        if os.path.isdir(os.path.join('image/train', label)):#/Users/mo3n/Documents/project/dog/git2/project_dog/
            # 각 라벨 디렉터리 내의 이미지 파일에 대해 반복합니다.
            for img_file in os.listdir(os.path.join('image/train',#/Users/mo3n/Documents/project/dog/git2/project_dog/
                                                     label)):
                # 이미지를 읽어옵니다.
                img = cv2.imread(os.path.join('image/train', #/Users/mo3n/Documents/project/dog/git2/project_dog/
                                              label, img_file)) 
                X.append(img) # 이미지 데이터를 X 리스트에 추가합니다.
                Y.append(label2id[label]) # 이미지의 라벨을 숫자 ID로 변환하여 Y 리스트에 추가합니다.
    return X, Y

# SIFT를 사용하여 이미지 특징 추출
def extract_sift_features(X):
    image_descriptors = [] #이미지 특징이 저장될 리스트

    # 이미지 추출기 생성
    sift = cv2.SIFT_create(nfeatures=200, nOctaveLayers=3, contrastThreshold=0.0005)
    # nfeatures : 검출최대특징수
    # nOctaveLayers=3 : 
    # ->nOctaveLayers: 이미지 피라미드에 사용할 계층 수 계층을 거듭할수록 이미지크기가 1/4(가로세로 각각 1/2)
    # contrastThreshold=0.0005 : 약한 특징에 대한 필터링 기준

    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i], None) #SIFT 특징추출
        # keypoints, descriptors = detector.detectAndCompute(image, mask,
        #             decriptors, useProvidedKeypoints)
        # image: 입력 이미지 
        # keypoints: 디스크립터 계산을 위해 사용할 특징점
        # descriptors(optional): 계산된 디스크립터
        # mask(optional): 특징점 검출에 사용할 마스크
        # useProvidedKeypoints(optional): True인 경우 특징점 검출을 수행하지 않음
        # ->특징점 검출과 특징 디스크립터 계산을 한 번에 수행
        image_descriptors.append(des) #추출된 특징을 리스트에 추가

    return image_descriptors

# K-means를 이용해 Bag of Words 딕셔너리 생성 함수 정의
# kmeans_bow 함수는 K-means 클러스터링을 사용하여 Bag of Words (BoW) 딕셔너리를 생성합니다.
# all_descriptors는 이미지 특징 디스크립터(특징 벡터)들을 포함하는 리스트입니다.
# ->image_descriptors의 집합
# num_clusters는 클러스터의 수를 나타냅니다.
# KMeans(n_clusters=num_clusters)는 클러스터의 수를 지정하여 K-means 클러스터링 모델을 초기화합니다.
# kmeans.fit(all_descriptors)는 주어진 특징 디스크립터 데이터에 K-means 클러스터링을 적용합니다.
# kmeans.cluster_centers_는 클러스터 중심을 나타내며, 이것이 Bag of Words 딕셔너리로 사용됩니다.
# bow_dict는 생성된 BoW 딕셔너리를 반환합니다.
def kmeans_bow(all_descriptors, num_clusters):
    # K-means 클러스터링을 수행하여 딕셔너리 생성
    kmeans = KMeans(n_clusters=num_clusters).fit(all_descriptors)  
    bow_dict = kmeans.cluster_centers_  # 클러스터 중심을 Bag of Words 딕셔너리로 사용
    return bow_dict


# create_features_bow 함수는 이미지의 특징 디스크립터(특징 벡터)를 Bag of Words (BoW) 형식으로 변환하는 역할을 합니다.
# image_descriptors는 이미지 특징 디스크립터(특징 벡터)들을 포함하는 리스트입니다.
# BoW는 Bag of Words 딕셔너리로, K-means 클러스터 중심을 의미합니다.
# num_clusters는 클러스터의 수를 나타냅니다.
# X_features는 변환된 특징 벡터를 저장하는 리스트입니다.
# 각 이미지에 대해 다음 작업을 수행합니다:
# features는 0으로 초기화된 특징 벡터를 생성합니다.
# 이미지의 특징 디스크립터가 None이 아닌 경우, 각 특징 디스크립터와 BoW 사이의 거리를 계산합니다.
# argmin은 가장 가까운 클러스터(BoW 단어)의 인덱스를 찾습니다.
# 해당 클러스터에 속한 특징을 하나씩 증가시켜 해당 클러스터의 특징 개수를 기록합니다.
# 변환된 특징 벡터를 X_features 리스트에 추가하고, 이를 반환합니다.
# 이미지의 특징들을 벡터로 변환 함수 정의

def create_features_bow(image_descriptors, BoW, num_clusters):
    # image_descriptors : 특징추출기
    X_features = []  # 이미지 특징 벡터를 저장할 리스트
    # 회전과 크기변환에도 불변한 특징을 벡터로 저장
    for i in range(len(image_descriptors)):
        features = np.array([0] * num_clusters)  # 0으로 초기화된 특징 벡터 생성

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)  # 클러스터 중심과의 거리 계산
            argmin = np.argmin(distance, axis=1)  # 가장 가까운 클러스터 인덱스 찾기
            for j in argmin:
                features[j] += 1  # 해당 클러스터에 포함된 특징 개수 증가
        X_features.append(features)  # 특징 벡터를 리스트에 추가합니다.
    return X_features

# 메인 실행 함수
def main():
    start = time.time()  # 시간 측정을 위한 시작 시간 기록
   
    # 라벨을 숫자 ID로 변환
    # 경로 확인 필요
    path = "image/train"
    file_list = os.listdir(path)
    label2id = {}

    idx = 0
    for i, label in enumerate(file_list): #file_list = image 하위 폴더 내역#/Users/mo3n/Documents/project/dog/git2/project_dog/
        if label == ".DS_Store": #디렉터리에 .DS_Store는 제외
            continue
        label2id[label] = idx #dictionary룰 통해
        idx += 1

    # 학습 데이터 읽기
    X, Y = read_data(label2id)  # 이미지 데이터와 라벨 데이터를 읽어옵니다.
    
    # 이미지의 SIFT 특징 추출
    image_descriptors = extract_sift_features(X)  # 이미지에서 SIFT 특징을 추출합니다.

    # 모든 descriptors를 하나의 리스트로 만들기
    all_descriptors = []
    for descriptors in image_descriptors:
        if descriptors is not None:
            for des in descriptors:
                all_descriptors.append(des)

    num_clusters = 100  # 클러스터의 수

    # BOW 딕셔너리 존재 유무 체크 후 생성 또는 로드
    # 경로 확인 필요
    if not os.path.isfile('bow.pkl'):  # 저장된 BoW 딕셔너리 파일이 존재하지 않는 경우
        BoW = kmeans_bow(all_descriptors, num_clusters)  # K-means를 사용하여 BoW 딕셔너리 생성
        pickle.dump(BoW, open('bow.pkl', 'wb'))  # 생성된 BoW 딕셔너리를 파일에 저장
    else:
        BoW = pickle.load(open('bow.pkl', 'rb'))  # 저장된 BoW 딕셔너리 파일을 로드합니다.


    # 이미지 특징들을 벡터로 변환
    # 이미지 특징을 BoW 벡터로 변환합니다.
    X_features = create_features_bow(image_descriptors, BoW, num_clusters)  

    # 학습과 테스트 데이터로 분할
    X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=0.2,
                                                         random_state=42, shuffle=True)
    
    # SVM 모델 학습
    svm = sklearn.svm.SVC(C=100, probability=True)
    svm.fit(X_train, Y_train)

    # KNN 모델 학습
    knn = KNeighborsClassifier(n_neighbors=10, weights='distance', leaf_size=200, p=2)
    knn.fit(X_train, Y_train)

    # 테스트 이미지로 

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # -> 이부분 반복문으로 바꿔야할듯?
    
    # --test 디폴트값
    # parser.add_argument('--test', default='test_1.jpg'
    #                     , help='test image data')


    # img_test = histo_clahe('image/test/' + opt.test) 
    # # -> 이미지를 histo_clache에 적용하고 사이즈, 컬러채널에 대한 변동사항을 적용한 결과를 반환

    # img = [img_test]
    # img_sift_feature = extract_sift_features(img)  # 테스트 이미지의 SIFT 특징을 추출합니다.
    # img_bow_feature = create_features_bow(img_sift_feature
    #                                     , BoW, num_clusters)  # 테스트 이미지의 BoW 특징을 생성합니다.

    # # SVM과 KNN을 사용한 예측
    # img_predict = svm.predict(img_bow_feature)  # SVM을 사용하여 이미지를 예측합니다.
    # img_predict2 = knn.predict(img_bow_feature)  # KNN을 사용하여 이미지를 예측합니다.

    # # 예측 확률
    # svm_prob = svm.predict_proba(img_bow_feature)[0][img_predict[0]]  # SVM 예측의 확률을 계산합니다.
    # knn_prob = knn.predict_proba(img_bow_feature)[0][img_predict2[0]]  # KNN 예측의 확률을 계산합니다.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 테스트 이미지 파일이 들어 있는 디렉토리 경로
# 경로 확인 필요
    test_dir = './image/test/0'

    # 디렉토리 내의 모든 파일에 대해 반복

    for filename in os.listdir(test_dir):
        if 'DS_Store' not in filename:
            # if filename.endswith('.jpg'):  # 이미지 파일인 경우에만 처리
            # 파일의 전체 경로를 생성
            img_path = os.path.join(test_dir, filename)# +'/' + onlyfiles

            print('\n',img_path,'\n')
            # 이미지를 histo_clahe에 적용하고 사이즈, 컬러 채널에 대한 변동사항을 적용한 결과 반환
            img_test = histo_clahe(img_path)

            # 이미지를 리스트에 추가
            img = [img_test]

            # 테스트 이미지의 SIFT 특징을 추출
            img_sift_feature = extract_sift_features(img)

            # 테스트 이미지의 BoW 특징을 생성
            img_bow_feature = create_features_bow(img_sift_feature, BoW, num_clusters)

            # SVM과 KNN을 사용한 예측
            img_predict = svm.predict(img_bow_feature)  # SVM을 사용하여 이미지를 예측
            img_predict2 = knn.predict(img_bow_feature)  # KNN을 사용하여 이미지를 예측

    # ------------------------------------------------------------------
            # 예측 확률
            # svm_prob = svm.predict_proba(img_bow_feature)[0][img_predict[0]]  # SVM 예측의 확률 계산
            # knn_prob = knn.predict_proba(img_bow_feature)[0][img_predict2[0]]  # KNN 예측의 확률 계산
            svm_prob = svm.predict_proba(img_bow_feature)[0][img_predict[0]]  # SVM 예측의 확률 계산
            knn_prob = knn.predict_proba(img_bow_feature)[0][img_predict2[0]]  # KNN 예측의 확률 계산

        # 각 이미지에 대한 결과를 여기에서 사용하거나 저장할 수 있습니다.

        # 예측 라벨

            for key, value in label2id.items():
                if value == img_predict[0]: #SVM 예측확률
                    svm_k = key
                if value == img_predict2[0]: #KNN 예측확률
                    knn_k = key

            # 예측 결과 문자열 생성
            df = pd.read_csv('DB.csv', encoding = 'cp949', header = 0)
            if (svm_prob < 0.65 and knn_prob < 0.55) or svm_k != knn_k:  # 예측이 불확실하거나 SVM과 KNN의 예측이 다른 경우 "미등록강아지"로 설정
                while True:
                    answer = input('이 강아지는 아직 등록되어 있지 않아요! 등록하시겠습니까? (Y/N)')
                    if answer != 'y' and answer !='Y' and answer != 'n' and answer != 'N':
                        print('올바른 입력값이 아닙니다.')
                        continue
                    else:
                        break
                if answer == 'n' or answer == 'N':
                        print('그러시던지')
                elif answer == 'y' or answer == 'Y':
                        while True:
                            move = input('사진을 등록하셨습니까? (Y/N)')
                            if move != 'y' and move !='Y' and move != 'n' and move != 'N':
                                print('올바른 입력값이 아닙니다.')
                                continue
                            else:
                                break
                        if move == 'Y' or move == 'y':
                            os.system('python register/YOLOv5/detect.py --source image/register_input/0 --weights register/YOLOv5/best.pt --option register --save-conf')
                            time.sleep(0.5)
                            os.system('python register/preprocess.py --savedir image/train')
                break                
            # elif svm_k==knn_k and max([knn.score(X_test, Y_test), svm.score(X_test, Y_test)]) >= 0.85:
            #     # (svm_prob < 0.65 and knn_prob < 0.55) or svm_k != knn_k의 반대 + accuracy도 활용?
            #     key = int(svm_k)
            #     print('이 강아지는 등록이 되어있는 강아지 입니다!' + '\n' + '제 이름은 {}, {}살 입니다! {}님과 함께 살고 있어요!'.format(df['강아지이름'][df['일련번호']==key].values[0], df['나이'][df['일련번호']==key].values[0], df['견주이름'][df['일련번호']==key].values[0]) +'\n' + '저는 {}예요 :3'.format(df['견종'][df['일련번호']==key].values[0]))
            #     break
            # else:
            #     break
                
            # 최고 정확도 출력
            #if svm.score(X_test, Y_test) > knn.score(X_test, Y_test):
            #    result += str(svm.score(X_test, Y_test))  # SVM 모델의 정확도 출력
            #else:
            #    result += str(knn.score(X_test, Y_test))  # KNN 모델의 정확도 출력
            # print(result)
            #print('\n',img_path,'\n','svm_score : ',svm_prob, 'knn_score : ', knn_prob)  # 결과 문자열 출력
            # print('\n',img_path,'\n' ,sum(img_sift_feature[0].flatten()))


if __name__ == "__main__":
    main()
