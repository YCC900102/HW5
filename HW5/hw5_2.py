import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import shutil
import os

# 清除舊的 TensorBoard 日誌
log_dir = './logs'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)


# 1. 載入 MNIST 資料集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. 資料預處理
# 正規化像素值到 0-1 之間
X_train = X_train / 255.0
X_test = X_test / 255.0

# 將數據重塑為適合 CNN 模型的輸入形狀
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-Hot 編碼標籤
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 3. 建立 Dense NN 模型
def create_dense_nn():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 4. 建立 CNN 模型
def create_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 5. 訓練 Dense NN 模型
dense_model = create_dense_nn()
tensorboard_dense = TensorBoard(log_dir='./logs/hw5_2_dense')
dense_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), 
                callbacks=[tensorboard_dense, EarlyStopping(monitor='val_loss', patience=5)])

# 6. 訓練 CNN 模型
cnn_model = create_cnn()
tensorboard_cnn = TensorBoard(log_dir='./logs/hw5_2_cnn')
cnn_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), 
              callbacks=[tensorboard_cnn, EarlyStopping(monitor='val_loss', patience=5)])

# 7. 模型評估
dense_loss, dense_accuracy = dense_model.evaluate(X_test, y_test)
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test)

print(f"Dense NN Test Accuracy: {dense_accuracy:.4f}")
print(f"CNN Test Accuracy: {cnn_accuracy:.4f}")

# 8. TensorBoard 指令
print("啟動 TensorBoard：執行以下指令")
print("tensorboard --logdir=./logs")
