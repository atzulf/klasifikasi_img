import streamlit as st
import numpy as np
import tensorflow as tf
import keras

print(tf.__version__)
print(keras.__version__)

from tensorflow.keras.models import load_model  # Menggunakan dari tensorflow.keras
from PIL import Image

# Memuat model CNN yang telah dilatih
model = load_model('cat_dog.h5')

# Definisikan label kelas
class_names = ['Cat', 'Dog']

# Fungsi untuk memproses gambar
def preprocess_image(image):
    # Ubah ukuran gambar menjadi (224, 224)
    image = image.resize((150, 150))
    # Konversi gambar menjadi array dan normalisasi
    image_array = np.array(image) / 255.0
    # Tambahkan dimensi batch
    return np.expand_dims(image_array, axis=0)

# Judul aplikasi
st.title("Cat vs Dog Image Classifier")

# Mengunggah gambar menggunakan sidebar
uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membuka dan menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Praproses gambar
    processed_image = preprocess_image(image)

    # Prediksi menggunakan model CNN
    predictions = model.predict(processed_image)
    pred_class = np.argmax(predictions[0])  # Mengambil indeks kelas dengan probabilitas tertinggi
    confidence = np.max(predictions[0]) * 100  # Mengambil tingkat kepercayaan prediksi

    # Menampilkan hasil prediksi
    st.header(f"Prediction: **{class_names[pred_class]}**")
    st.header(f"Confidence: **{confidence:.2f}%**")
    
