import cv2
import joblib
import numpy as np 

# Muat model KNN dan scaler
knn = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 🔄 Konversi frame dari BGR ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Ambil pixel tengah gambar (dalam RGB)
    height, width, _ = frame_rgb.shape
    pixel_center = frame_rgb[height//2, width//2]
    
    # Normalisasi pixel sebelum prediksi
    pixel_center_scaled = scaler.transform([pixel_center])
    
    # Prediksi warna
    color_pred = knn.predict(pixel_center_scaled)[0]
    
    # Gambar titik di tengah frame
    cv2.circle(frame, (width//2, height//2), 5, (0, 255, 0), -1)
    
    # Tampilkan warna pada frame
    cv2.putText(frame, f'Color: {color_pred}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.imshow('Color Detection (Fixed RGB)', frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
