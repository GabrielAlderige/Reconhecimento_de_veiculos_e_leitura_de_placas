📘 Detector de Veículos e Placas com YOLOv8 + OCR

Autor: Gabriel Alderige

Este projeto realiza detecção de veículos, detecção de placas e leitura automática (OCR) em tempo real usando a webcam.
As placas reconhecidas são salvas automaticamente em um arquivo CSV com data e hora.

O sistema utiliza:

YOLOv8n → detecção de veículos

Modelo personalizado (best.pt) → detecção de placas

EasyOCR → leitura do texto da placa

OpenCV → captura da webcam e exibição

CSV → registro das placas detectadas com cooldown para evitar duplicados

🧠 Tecnologias Utilizadas

Python 3

Ultralytics YOLOv8

OpenCV

EasyOCR

Torch

CSV

Regex (limpeza de texto)

📂 Estrutura do Projeto
/meu_projeto
│── detector.py # Script principal (detecção + OCR + CSV)
│── yolov8n.pt # Modelo YOLO pré-treinado COCO
│── best.pt # Modelo YOLO treinado para detectar placas
│── requirements.txt # Dependências
│── README.md # Documentação
└── placas_detectadas.csv # Gerado automaticamente ao rodar

🚀 Como Rodar o Projeto

Siga estas etapas para rodar o sistema completo de detecção de veículos + placas + OCR + salvamento em CSV.

✅ 1️⃣ Pré-requisitos

Certifique-se de ter instalado:

Python 3.10 ou superior

Pip atualizado

Webcam funcional

Modelos YOLO:

yolov8n.pt

best.pt (modelo treinado para placas)

✅ 2️⃣ Criar e ativar ambiente virtual
Windows
python -m venv venv
venv\Scripts\activate

Linux / MacOS
python3 -m venv venv
source venv/bin/activate

Quando estiver ativo, aparecerá assim:

(venv) C:\seu\projeto>

✅ 3️⃣ Instalar as dependências

Com o ambiente virtual ativo e na pasta do projeto:

pip install -r requirements.txt

Ou, manualmente:

pip install ultralytics opencv-python easyocr torch torchvision

✅ 4️⃣ Executar o programa

Para iniciar o detector:

python detector.py

A câmera abrirá e você verá:

Veículos detectados → caixa verde

Placas detectadas → caixa azul

OCR rodando em tempo real

Placas registradas automaticamente no CSV

🔄 5️⃣ Se a webcam não abrir

Se aparecer:

Erro: não foi possível acessar a câmera.

Troque a linha:

cap = cv2.VideoCapture(0)

Por:

cap = cv2.VideoCapture(1)

🎯 Funcionalidades

✔ Detecção em tempo real
✔ Veículos marcados com caixa verde
✔ Placas marcadas com caixa azul
✔ Exibição da confiança do modelo
✔ Suporte completo ao YOLOv8
✔ OCR automático com EasyOCR
✔ Registro automático em CSV

🧠 Classes de veículos detectadas (COCO)

2 → Carro

3 → Moto

5 → Ônibus

7 → Caminhão

👤 Autor

Gabriel Alderige
