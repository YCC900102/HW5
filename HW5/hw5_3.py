import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import os
import shutil

# 清除舊的 TensorBoard 日誌
log_dir = './logs'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

# 1. 載入 CIFAR-10 資料集
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 2. 資料預處理
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-Hot 編碼標籤
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 資料增強 (Data Augmentation)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# 3. 建立 VGG16 模型
base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model_vgg16.trainable = False

model_vgg16 = Sequential([
    base_model_vgg16,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model_vgg16.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 建立 VGG19 模型
base_model_vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model_vgg19.trainable = False

model_vgg19 = Sequential([
    base_model_vgg19,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model_vgg19.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 5. 訓練 VGG16 模型
tensorboard_vgg16 = TensorBoard(log_dir='./logs/vgg16')
model_vgg16.fit(datagen.flow(X_train, y_train, batch_size=64),
                epochs=50, validation_data=(X_test, y_test),
                callbacks=[tensorboard_vgg16, EarlyStopping(monitor='val_loss', patience=5)])

# 6. 訓練 VGG19 模型
tensorboard_vgg19 = TensorBoard(log_dir='./logs/vgg19')
model_vgg19.fit(datagen.flow(X_train, y_train, batch_size=64),
                epochs=50, validation_data=(X_test, y_test),
                callbacks=[tensorboard_vgg19, EarlyStopping(monitor='val_loss', patience=5)])

# 7. 模型評估
loss_vgg16, acc_vgg16 = model_vgg16.evaluate(X_test, y_test)
loss_vgg19, acc_vgg19 = model_vgg19.evaluate(X_test, y_test)

print(f"VGG16 Test Accuracy: {acc_vgg16:.4f}")
print(f"VGG19 Test Accuracy: {acc_vgg19:.4f}")

# 8. TensorBoard 指令
print("啟動 TensorBoard：執行以下指令")
print("tensorboard --logdir=./logs")