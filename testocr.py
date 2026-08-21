import cv2
import math
import numpy as np
from ultralytics import YOLO
import time
import easyocr  # <-- NOWOŚĆ

# ==========================================
# KONFIGURACJA TESTOWA I SPRZĘTOWA
# ==========================================
MODEL_PATH = "best.pt"  
ACTIVE_MISSION = 2      
TEAM_COLOR = "red"      
TEST_SOURCE = 0     

# [WAŻNE DLA JETSONA]: 
# Na laptopie z Intelem zostaw False. 
# Kiedy wgrasz to na Jetsona, zmień na True, aby EasyOCR użył rdzeni CUDA i przyspieszył 10-krotnie!
USE_GPU_OCR = True 

print("Ładowanie modelu OCR (EasyOCR)...")
OCR_READER = easyocr.Reader(['en'], gpu=USE_GPU_OCR) 

CLASS_NAMES = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'panel', 11: 'empty'
}
MISSION1_SCORES = {"Rifle Soldier": 1, "Tank": 7, "Helicopter": 8}

# ==========================================
# LOGIKA WIZYJNA
# ==========================================
def assemble_target_number(clean_frame, detections, model, team_color="blue"):
    panels = [d for d in detections if d["class"] == "panel"]
    digits = [d for d in detections if str(d["class"]).isdigit()]
    
    if not panels or len(digits) < 2: 
        return None

    panels.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    panel = panels[0]
    px1, py1, px2, py2 = panel["bbox"]
    
    # 1. WYCINANIE
    pad = 80
    h, w = clean_frame.shape[:2]
    x1_cv, y1_cv = max(0, int(px1) - pad), max(0, int(py1) - pad)
    x2_cv, y2_cv = min(w, int(px2) + pad), min(h, int(py2) + pad)
    
    if x2_cv - x1_cv < 10 or y2_cv - y1_cv < 10: return None
    crop = clean_frame[y1_cv:y2_cv, x1_cv:x2_cv]
    
    # 2. SZUKANIE GROTA NA PODSTAWIE CYFR Z PIERWSZEGO SKANU
    digits.sort(key=lambda x: x.get("confidence", 1.0), reverse=True)
    d1, d2 = digits[:2]
    
    c1_x, c1_y = (d1["bbox"][0] + d1["bbox"][2]) / 2, (d1["bbox"][1] + d1["bbox"][3]) / 2
    c2_x, c2_y = (d2["bbox"][0] + d2["bbox"][2]) / 2, (d2["bbox"][1] + d2["bbox"][3]) / 2
    
    base_x = ((c1_x + c2_x) / 2) - x1_cv
    base_y = ((c1_y + c2_y) / 2) - y1_cv
    base_point = np.array([base_x, base_y])
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if team_color == "red":
        m1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
    else:
        mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([140, 255, 255]))
        
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    main_contour = max(contours, key=cv2.contourArea)
    pts = main_contour.reshape(-1, 2)
    
    distances = np.linalg.norm(pts - base_point, axis=1)
    tip_local = pts[np.argmax(distances)]
    
    vec_up = tip_local - base_point
    
    # 3. PROSTOWANIE OBRAZKA
    angle_rad = math.atan2(vec_up[1], vec_up[0])
    angle_deg = math.degrees(angle_rad)
    rotation_angle = angle_deg + 90 
    
    h_c, w_c = crop.shape[:2]
    M_rot = cv2.getRotationMatrix2D((w_c // 2, h_c // 2), rotation_angle, 1.0)
    aligned_crop = cv2.warpAffine(crop, M_rot, (w_c, h_c), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    
    cv2.imshow("3. WYPROSTOWANY PANEL", aligned_crop)
    
    # ==========================================
    # 4. DRUGI ETAP: EASYOCR (Szybki odczyt cyfr)
    # ==========================================
    # Wymuszamy TYLKO cyfry, co drastycznie przyspiesza działanie modelu
    ocr_results = OCR_READER.readtext(aligned_crop, allowlist='0123456789')
    
    if not ocr_results:
        return None
        
    # EasyOCR zwraca listę w formacie: [ ([[x,y],...], 'tekst', pewność), ... ]
    # Krok A: Sortujemy wyniki po osi X (od lewej do prawej), na wypadek gdyby odczytał cyfry osobno
    ocr_results.sort(key=lambda x: x[0][0][0])
    
    # Krok B: Łączymy cały znaleziony tekst w jeden ciąg
    combined_text = "".join([res[1] for res in ocr_results])
    
    if len(combined_text) >= 2:
        final_number = int(combined_text[:2]) # Bierzemy twardo pierwsze dwie cyfry
        global_center = np.array([base_point[0] + x1_cv, base_point[1] + y1_cv])
        print(f"-> [OCR SUKCES] Odczytano liczbę: {final_number}")
        return final_number, global_center
        
    return None

# ==========================================
# GŁÓWNA PĘTLA TESTOWA
# ==========================================
def main():
    print(f"Ładowanie modelu: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(TEST_SOURCE)
    
    if not cap.isOpened():
        print("Błąd: Nie można otworzyć źródła wideo.")
        return

    print("Kamera aktywna. Wciśnij 'q', aby wyjść.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Koniec strumienia wideo.")
            break
        
        clean_frame = frame.copy()
        start_time = time.perf_counter()
        # 1. Inferencja YOLO (Skan główny)
        results = model.predict(frame, imgsz=640, conf=0.4, verbose=False, device="cpu")
        raw_detections = []
        
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                if float(box.conf[0]) < 0.4: continue
                class_id = int(box.cls[0])
                cx, cy, w, h = box.xywh[0].tolist()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                cls_name = CLASS_NAMES.get(class_id, str(class_id))
                raw_detections.append({
                    "class": cls_name, "class_id": class_id,
                    "x": cx, "y": cy, "bbox": [x1, y1, x2, y2]
                })
                
                # Rysowanie ramek TYLKO debugowania z pierwszego skanu
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, cls_name, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 2. Logika przetwarzania
        processed = []
        if raw_detections:
            if ACTIVE_MISSION == 1:
                for d in raw_detections:
                    if d["class"] in MISSION1_SCORES:
                        processed.append({"class": d["class"], "score": MISSION1_SCORES[d["class"]], "x": d["x"], "y": d["y"]})
            elif ACTIVE_MISSION == 2:
                if any(d["class"] == "panel" for d in raw_detections):
                    result = assemble_target_number(clean_frame, raw_detections, model=model, team_color=TEAM_COLOR)
                    if result:
                        number, coords = result
                        processed.append({"class": "NumberedPanel", "number": number, "x": float(coords[0]), "y": float(coords[1])})
                        cv2.circle(frame, (int(coords[0]), int(coords[1])), 8, (0, 0, 255), -1)
                        cv2.putText(frame, f"WYNIK: {number}", (int(coords[0])+10, int(coords[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 3. Symulacja MQTT
        if processed:
            print(f"[MQTT SYMULACJA] -> {processed}")
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000  # Zamiana sekund na milisekundy
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        # Rysujemy ładny, żółty tekst w lewym górnym rogu
        cv2.putText(frame, f"Czas: {processing_time_ms:.1f} ms | FPS: {fps:.1f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # 4. Wyświetlanie obrazu
        cv2.imshow("Test Detekcji Laptop", frame)
        
        if isinstance(TEST_SOURCE, str) and TEST_SOURCE.lower().endswith(('.jpg', '.jpeg', '.png')):
            wait_time = 0
        else:
            wait_time = 1
            
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()