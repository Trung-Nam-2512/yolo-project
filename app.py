import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO
import uuid
import shutil

# Load mô hình YOLO
model = YOLO('best.pt')  # Đảm bảo file best.pt cùng thư mục hoặc cập nhật đường dẫn

st.title("🦟 Nhận diện côn trùng gây hại lúa")

uploaded_file = st.file_uploader("📤 Tải ảnh lên", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lưu ảnh tạm thời
    img_id = str(uuid.uuid4())
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    image_path = os.path.join(temp_dir, img_id + ".jpg")

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Hiển thị ảnh gốc
    st.image(image_path, caption="Ảnh gốc", use_container_width=True)

    # Dự đoán
    st.write("🔍 Đang nhận diện...")
    results = model.predict(source=image_path, save=True, conf=0.25)

    # Đường dẫn ảnh kết quả (YOLO tự lưu vào runs/detect/predict)
    result_dir = results[0].save_dir
    result_img = os.path.join(result_dir, os.path.basename(image_path))

    # Hiển thị kết quả
    st.image(result_img, caption="📌 Kết quả nhận diện", use_container_width=True)

    # Xóa thư mục tạm sau khi hiển thịstreamlit run app.py

    shutil.rmtree(temp_dir)
