from ultralytics import YOLO
import cv2
import torch
import os

# Verifica se há GPU disponível
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carregar o modelo treinado
model_path = "Teste/testeModel"  # Substitua pelo caminho do seu modelo
model = YOLO(model_path).to(device)  # Move para GPU se disponível

def detect_image(image_path):
    """Realiza a detecção em uma imagem e exibe o resultado."""
    img = cv2.imread(image_path)
    results = model(image_path) 
    
    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf < 0.5:  # Filtrar detecções com baixa confiança
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]}: {conf:.2f}"
            
            # Desenhar a bounding box e o label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
    
    # Ajustar escala dinamicamente para exibição
    max_size = 800
    h, w = img.shape[:2]
    scale = max_size / max(h, w) if max(h, w) > max_size else 1
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h))
    
    cv2.imshow("Detection", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_all_images_in_folder(folder_path):
    """Aplica a detecção em todas as imagens da pasta."""
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            print(f"Detectando: {filename}")
            detect_image(image_path)

# Caminho da pasta com as imagens para teste
folder_path = "Teste/testeImages"  # Substitua pelo caminho da pasta

detect_all_images_in_folder(folder_path)