import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import io
from PIL import Image

# Load and preprocess data
@st.cache_data
def load_data():
    train_df = pd.read_csv("mnist_test.csv")
    test_df = pd.read_csv("mnist_train.csv")
    return train_df, test_df

@st.cache_data
def preprocess_data(train_df, test_df):
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]
    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler


def train_model(model_name, X_train, y_train):
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = SVC(kernel="linear")

    model.fit(X_train, y_train)
    return model

# hiển thị dự đoán mẫu
def show_predictions(X_test, y_test, y_pred):
    st.subheader("Sample Predictions")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        img = X_test[i].reshape(28, 28)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"True: {y_test.iloc[i]}\nPred: {y_pred[i]}")
        ax.axis("off")
    st.pyplot(fig)

# dự đoán hình ảnh do người dùng tải lên
def predict_uploaded_image(image, model, scaler):
    image = image.convert("L").resize((28, 28))
    img_array = np.array(image).reshape(1, -1)
    img_scaled = scaler.transform(img_array)
    prediction = model.predict(img_scaled)
    return prediction[0]

# tiêu đề
st.title("MNIST Handwritten Digit Classification")
st.write("Phân loại chữ số viết tay sử dụng Random Forest hoặc SVM.")

model_choice = st.selectbox("Chọn mô hình:", ["Random Forest", "SVM (Linear)"])

train_df, test_df = load_data()
X_train, y_train, X_test, y_test, scaler = preprocess_data(train_df, test_df)

# Giới hạn dữ liệu để tăng tốc nếu cần
if model_choice == "SVM (Linear)":
    X_train = X_train[:10000]
    y_train = y_train[:10000]

model = train_model(model_choice, X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=range(10), yticklabels=range(10), ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("True")
st.pyplot(fig_cm)

show_predictions(X_test, y_test, y_pred)

# Upload and predict custom image
st.subheader("Dự đoán ảnh chữ số tải lên")
uploaded_file = st.file_uploader("Tải ảnh PNG của chữ số viết tay (28x28 pixel) lên:", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã tải lên", width=150)
    prediction = predict_uploaded_image(image, model, scaler)
    st.success(f"Dự đoán: {prediction}")

# Download classification report
st.subheader("Tải báo cáo")
report_text = classification_report(y_test, y_pred)
report_bytes = io.BytesIO()
report_bytes.write(report_text.encode())
report_bytes.seek(0)
st.download_button(label="Tải báo cáo phân loại", data=report_bytes, file_name="classification_report.txt")
