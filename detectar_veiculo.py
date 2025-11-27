import cv2
from ultralytics import YOLO
import torch
import easyocr
import csv
import time
import os
import re

# OCR
reader = easyocr.Reader(['en'])  # Inglês funciona bem para placas

# Desativar gradientes (acelera)
torch.set_grad_enabled(False)

# === CARREGAR MODELOS ===
vehicle_model = YOLO("yolov8n.pt")  # COCO
plate_model = YOLO("best.pt")       # Modelo de placas

vehicle_classes = [2, 3, 5, 7]  # car, motorbike, bus, truck

# Configurações
VEHICLE_CONF = 0.4
PLATE_CONF = 0.5

# CSV
CSV_FILE = "placas_detectadas.csv"

# Criar CSV se não existir
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["placa", "timestamp"])

# Cooldown para evitar salvar repetidas
last_detected = {}
COOLDOWN = 3  # segundos


def save_plate(plate_text):
    plate = plate_text.strip()

    # remover caracteres inválidos
    plate = re.sub(r"[^A-Z0-9]", "", plate.upper())

    # ignorar se a placa vier muito curta
    if len(plate) < 5:
        return

    current_time = time.time()

    # cooldown anti-duplicado
    if plate in last_detected and current_time - last_detected[plate] < COOLDOWN:
        return

    last_detected[plate] = current_time

    # Salvar CSV
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([plate, time.strftime("%Y-%m-%d %H:%M:%S")])

    print(f"[✔] Placa salva: {plate}")


def draw_label(img, text, x, y, color):
    cv2.putText(img, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)


# Abrir câmera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: não foi possível acessar a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame.")
        break

    # DETECTAR VEÍCULOS
    vehicle_results = vehicle_model.predict(frame, conf=VEHICLE_CONF, stream=True, verbose=False)

    for result in vehicle_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = vehicle_model.names[cls_id]

            # Desenhar veículo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_label(frame, f"{label} {conf:.2f}", x1, y1, (0, 255, 0))

            # RECORTE DO VEÍCULO
            vehicle_roi = frame[y1:y2, x1:x2]
            if vehicle_roi.size == 0:
                continue

            # DETECTAR PLACA
            plate_results = plate_model.predict(vehicle_roi, conf=PLATE_CONF, verbose=False)

            for p_result in plate_results:
                for p_box in p_result.boxes:
                    px1, py1, px2, py2 = map(int, p_box.xyxy[0])

                    fx1 = x1 + px1
                    fy1 = y1 + py1
                    fx2 = x1 + px2
                    fy2 = y1 + py2

                    plate_label = plate_model.names[int(p_box.cls[0])]
                    plate_conf = float(p_box.conf[0])

                    # Desenhar placa
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
                    draw_label(frame, f"{plate_label} {plate_conf:.2f}", fx1, fy1, (255, 0, 0))

                    # ======== OCR ========
                    plate_roi = frame[fy1:fy2, fx1:fx2]

                    if plate_roi.size > 0:
                        ocr_result = reader.readtext(plate_roi)

                        if ocr_result:
                            text = ocr_result[0][1]  # apenas o melhor texto
                            save_plate(text)


    cv2.imshow("Detecção de Veículos, Placas e OCR", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
