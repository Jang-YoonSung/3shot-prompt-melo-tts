## Introduction

### Fast API를 사용한 Docker 기반 llm 서비스

**1. text generation using llama3 - few shot prompting**

- stt를 통한 음성파일 요약에서 모르는 단어가 나올 경우 성능이 떨어진다는 문제점을 통해 고안한 방법
- 허깅페이스의 라마3 한국어 파인튜닝 모델을 사용
- 파인튜닝 없이 간단한 몇개의 예시만으로 모델의 성능이 크게 향상

**2. Melo TTS 서비스 (OpenSource model)**

- 모델이 모르는 사내 약어나 전문적인 단어를 generation 모델로 생성 후, tts를 통해 wav파일로 전송
- back-end를 통해 stt 학습 서버로 전송

**3. OpenVoice 목소리 변경 시스템 (OpenSource)**

- Melo TTS 개발자들이 해당 모델의 Fine-tuning에 불편함과 에러사항이 많다는 점을 참고하여 만든 서비스
- 원하는 음성 파일을 입력하면 해당 음성을 분석하여 기본 음성으로 만들어진 wav 파일에 음성을 튜닝

**4. 도커 기반 환경 구성**

- docker hub의 엔비디아 쿠다 이미지를 사용하여 원하는 쿠다 버전과, 파이썬 버전을 선택할 수 있게 이미지를 구성
- MeloTTS와의 의존성 검증
