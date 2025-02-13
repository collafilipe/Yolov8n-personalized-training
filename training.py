from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="Teste/testeYaml", # Caminho para o arquivo YAML
        epochs=100,  # Aumente conforme necessário
        batch=4,  # Tente 16 ou 32 para uma boa performance
        device="0",  # Se você tiver uma GPU, use 'cuda' (se não, 'cpu')
        name="treinamento",  # Nome
        project="resultados",
        workers=1,  # Use mais workers para melhorar o carregamento dos dados
        lr0=0.01,  # Taxa de aprendizado inicial
        momentum=0.937,  # Momento padrão
        weight_decay=0.0005,  # Regularização de peso para evitar overfitting
        warmup_epochs=3,  # Use um pequeno número de épocas para aquecer o modelo no início
    )
