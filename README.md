# 🎯 3shot-Prompt-Melo-TTS

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)
![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

FastAPI 기반 **키워드 기반 텍스트 생성 & 음성 합성 서비스**입니다.  
Llama3 모델과 Melo-TTS를 활용하며, 3-shot prompting을 적용하여 텍스트 품질을 향상시킨 폐쇄망 서비스입니다.

---

## 🚀 Features
- ✍️ **Text Generation API**: 키워드 기반 문장 생성 (모델: Llama3)
- 🔊 **TTS (Text-to-Speech)**: 한국어 텍스트 → 자연스러운 음성 합성 (Melo-TTS)
- 🐳 **Docker Compose 지원**: 로컬/서버 환경 어디서든 빠른 실행 가능
- 📡 **Swagger UI**: 직관적인 API 테스트 및 문서화

---

## 📂 Project Structure
```bash
3shot-prompt-melo-tts/
├── openvoice            # 목소리 복제 라이브러리
├── router/              # API 라우터
│   └── db.py
├── service/             # API 서비스
│   └── db_service.py    
├── main.py
├── Dockerfile  
├── docker-compose.yml
├── test.py              # 폐쇠망 및 서비스 테스트
├── requirements.txt
└── README.md
```

## 🏗 System Architecture
```bash
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|   FastAPI Server  +------>+   Llama Model     +------>+   Melo-TTS Model  |
|                   |       |                   |       |                   |
+-------------------+       +-------------------+       +-------------------+
```
* FastAPI Server: 클라이언트로부터 키워드를 받아 Llama 모델에 전달

* Llama Model: 키워드를 기반으로 문장 생성

* Melo-TTS Model: 생성된 문장을 음성으로 변환

## ⚙️ Installation & Usage

1️⃣ Clone Repository
```bash
git clone https://github.com/Jang-YoonSung/3shot-prompt-melo-tts.git
cd 3shot-prompt-melo-tts
```
2️⃣ Setup Environment

```.env``` 파일 생성:
```bash
# 환경 변수 설정 (필요 시 추가)
```
3️⃣ Run with Docker Compose
```bash
docker-compose up --build
```
서비스 실행 후 Swagger UI에서 API 테스트 가능:

👉 http://localhost:8000/docs

## 🛠 Tech Stack
* Python - Programming Language

* FastAPI - Web Framework

* Llama3 - Text Generation Model

* Melo-TTS - Text-to-Speech Model (한국어)

* Docker - Containerization

## 📄 License
MIT License

© 2025 Jang-YoonSung
