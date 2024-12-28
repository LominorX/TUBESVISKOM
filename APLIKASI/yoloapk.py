import os
import cv2
from ultralytics import YOLO
import argparse
from flask import Flask, request, render_template, send_from_directory


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Memuat model YOLOv8
print("\n[INFO] Memuat model...")
model = YOLO("best.pt")

@app.route('/')
def index():
    return render_template('gojo.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Tidak ada file yang diunggah"

    file = request.files['file']
    if file.filename == '':
        return "Nama file kosong"

    if file and allowed_file(file.filename):
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, file.filename)
        
        file.save(input_path)
        
        # Memproses file
        if file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            process_image(model, input_path, output_path)
        else:
            return "Format file tidak didukung. Harap unggah gambar (.jpg, .png)."

        return send_from_directory(OUTPUT_FOLDER, file.filename, as_attachment=True)

    return "Terjadi kesalahan saat mengunggah file."

# Fungsi untuk memproses gambar
def process_image(model, input_path, output_path):
    print("[INFO] Memproses gambar...")
    
    # Membaca gambar
    img = cv2.imread(input_path)

    # Deteksi objek
    results = model(img)

    # Menampilkan dan menyimpan hasil deteksi
    result_img = results[0].plot()  # Menambahkan kotak dan label
    cv2.imwrite(output_path, result_img)
    print(f"[INFO] Hasil disimpan ke {output_path}")

# Fungsi untuk mengecek ekstensi file yang diperbolehkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == "__main__":
    app.run(debug=True)
