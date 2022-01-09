from PIL import Image
import keras
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 함수 부분!
def resize_images(img_path):  # 28*28이 아닌 사진파일들을 변환해 주는 함수입니다!
    images=glob.glob(img_path + "/*.jpg")  
    print(len(images), " images to be resized.")
    # 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
    target_size=(28,28)
    for img in images:
        old_img=Image.open(img)
        new_img=old_img.resize(target_size,Image.ANTIALIAS)
        new_img.save(img, "JPEG")
    print(len(images), " images resized.")


def load_data(img_path, number_of_data=300):  # x_train, y_train 에 데이터를 넣어주는 함수입니다!
    # 가위 : 0, 바위 : 1, 보 : 2
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)
    idx=0
    for file in glob.iglob(img_path+'/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1  
    
    for file in glob.iglob(img_path+'/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
    print("학습데이터의 이미지 개수는", idx,"입니다.")
    return imgs, labels


def set_model():  # 사용 모델을 설정하고 하이퍼 피라미터를 설정 합니다.
    model=keras.models.Sequential()
    model.add(keras.layers.Conv2D(256, (3,3), activation='relu', input_shape=(28,28,3)))
    model.add(keras.layers.MaxPool2D(2,2))
    model.add(keras.layers.Conv2D(512, (3,3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(65, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))
    return model


def edu_model(model, x_train, y_train):  # 선정한 모델으로 학습을 시작합니다.
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train,y_train, epochs=10)

def test_model(model, x_train, y_train):  # 학습이 잘 되었는지 다른종류의 가위바위보 사진을 통해 검증해 봅시다!(tesorflow 공식 가위바위보 이미지파일)
    test_loss, test_accuracy = model.evaluate(x_test,y_test, verbose=2)
    print("test_loss: {} ".format(test_loss))
    print("test_accuracy: {}".format(test_accuracy))

# 메인 부분!
# resize_images()
path = ["/aiffel/rock_scissor_paper/scissor", "/aiffel/rock_scissor_paper/rock", "/aiffel/rock_scissor_paper/paper",  # 가위 이미지가 저장된 디렉토리! 여러번 쓰기 싫어용~ (테스트 포함)
    "/aiffel/rock_scissor_paper/test/scissor", "/aiffel/rock_scissor_paper/test/rock", "/aiffel/rock_scissor_paper/test/paper"]
for path in path:
    print(path)
    image_dir_path = os.getenv("HOME") + path
    resize_images(image_dir_path)

# load_data()
image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper" # 일반 데이터
(x_train, y_train)=load_data(image_dir_path,3000)
x_train_norm = x_train/255.0   # 입력은 0~1 사이의 값으로 정규화
print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))

test_image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test" # 테스트용 데이터
(x_test, y_test)=load_data(test_image_dir_path,372)
x_test_norm = x_test/255.0   # 입력은 0~1 사이의 값으로 정규화
print("x_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))

# 시험용 이미지가 잘 들어왔는지 출력해 봅시다!
plt.imshow(x_train[0])
print('라벨: ', y_train[0])

# 검증용 테스트 이미지가 잘 들어왔는지도 출력해 봅시다!
plt.imshow(x_test[0])
print('라벨: ', y_train[0])

# set_model()
model = set_model()

# edu_model()
edu_model(model,x_train,y_train)

# test_model()
test_model(model,x_train,y_train)
