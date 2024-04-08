import streamlit as st
import cv2
import numpy as np

def detect_faces(image):
    # Haar Cascadeを使用して顔検出を行う
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def main():
    st.title("顔検出アプリ")
    st.write("画像をアップロードしてください")

    # 画像のアップロード
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # アップロードされた画像をOpenCVの形式に変換
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)

        # 顔検出を実行
        faces = detect_faces(image)

        # 顔をボックスで囲む
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # モザイクをかける処理
        mosaic_image = image.copy()
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (w//10, h//10))
            face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
            mosaic_image[y:y+h, x:x+w] = face

        # 処理結果を表示
        st.image(mosaic_image, channels="BGR", caption="Processed Image", use_column_width=True)

        # 解除ボタンを配置して、押下すると元画像を表示
        if st.button("解除"):
            st.image(image, channels="BGR", caption="Original Image", use_column_width=True)

if __name__ == "__main__":
    main()

