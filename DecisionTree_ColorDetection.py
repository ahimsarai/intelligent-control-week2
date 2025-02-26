import cv2
import joblib
import numpy as np


# Muat model Decision Tree dan scaler
dt_model = joblib.load('decision_tree_model.pkl')
scaler = joblib.load('scaler.pkl')


# 🔄 Ubah referensi warna ke RGB (bukan BGR)
color_reference = {
    'Red': np.array([255, 0, 0]),
    'Green': np.array([0, 255, 0]),
    'Blue': np.array([0, 0, 255]),
    'Yellow': np.array([255, 255, 0]),
    'Cyan': np.array([0, 255, 255]),
    'Magenta': np.array([255, 0, 255]),
    'Black': np.array([0, 0, 0]),
    'White': np.array([255, 255, 255]),
    'Gray': np.array([128, 128, 128])
}

def calculate_accuracy(predicted_color, pixel_rgb):
    """Hitung akurasi berdasarkan jarak warna Euclidean."""
    reference_rgb = color_reference.get(predicted_color, np.array([0, 0, 0]))
    distance = np.linalg.norm(reference_rgb - pixel_rgb)
    max_distance = np.linalg.norm(np.array([255, 255, 255]))  # Jarak maksimum di RGB
    accuracy = 100 * (1 - distance / max_distance)
    return max(0, min(accuracy, 100))  # Pastikan akurasi antara 0-100%


# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔄 Konversi frame dari BGR ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height, width, _ = frame.shape
    box_size = 50

    # Koordinat untuk dua bounding box
    box1_top_left = (width // 4 - box_size // 2, height // 2 - box_size // 2)
    box2_top_left = (3 * width // 4 - box_size // 2, height // 2 - box_size // 2)
    
    # Ambil pixel tengah dari masing-masing bounding box (dalam RGB)
    pixel_box1 = frame_rgb[box1_top_left[1] + box_size // 2, box1_top_left[0] + box_size // 2]
    pixel_box2 = frame_rgb[box2_top_left[1] + box_size // 2, box2_top_left[0] + box_size // 2]

    # Normalisasi pixel sebelum prediksi
    pixel_box1_scaled = scaler.transform([pixel_box1])
    pixel_box2_scaled = scaler.transform([pixel_box2])

    # Prediksi warna
    color_pred1 = dt_model.predict(pixel_box1_scaled)[0]
    color_pred2 = dt_model.predict(pixel_box2_scaled)[0]

    # Hitung akurasi prediksi
    acc1 = calculate_accuracy(color_pred1, pixel_box1)
    acc2 = calculate_accuracy(color_pred2, pixel_box2)

    # Gambar bounding box
    cv2.rectangle(frame, box1_top_left, (box1_top_left[0] + box_size, box1_top_left[1] + box_size), (255, 0, 0), 2)
    cv2.rectangle(frame, box2_top_left, (box2_top_left[0] + box_size, box2_top_left[1] + box_size), (0, 255, 0), 2)

    # Tampilkan warna dan akurasi di atas bounding box
    cv2.putText(frame, f'{color_pred1} ({acc1:.2f}%)', (box1_top_left[0], box1_top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f'{color_pred2} ({acc2:.2f}%)', (box2_top_left[0], box2_top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow('Color Detection with Accuracy (Fixed RGB)', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
