import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

# 1. 載入 Iris 資料集
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# 2. 資料預處理
# 標準化特徵
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-Hot 編碼標籤
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# 分割訓練與測試資料集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 3. 建立模型
model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# 4. 編譯模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. 設定 TensorBoard 回調函數
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/hw5_1', histogram_freq=1)

# 6. 訓練模型
history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                    validation_data=(X_test, y_test),
                    callbacks=[tensorboard_callback, tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])

# 7. 評估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# 8. TensorBoard 啟動指令
print("啟動 TensorBoard：執行以下指令")
print("tensorboard --logdir=./logs/hw5_1")
