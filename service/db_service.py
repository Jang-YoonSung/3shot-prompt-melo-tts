from fastapi import HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse

import logging
import os
import re
import zipfile
from io import BytesIO
from pathlib import Path

import sys
from melo.api import TTS

# open voice 관련 라이브러리
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# huggingface 관련 모델 라이브러리
from transformers import pipeline
import time
import logging
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,               # 4비트 양자화 활성화
)


logging.basicConfig(level=logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = "cuda:0" if torch.cuda.is_available() else "cpu"


model = AutoModelForCausalLM.from_pretrained(
    "MLP-KTLim/llama-3-Korean-Bllossom-8B",
    quantization_config=quantization_config,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B", use_fast=True)



logging.info(device)
pipe = pipeline(
    "text-generation",
    # model="meta-llama/Llama-3.2-1B-Instruct",
    model=model,
    tokenizer=tokenizer,
    # model="/app/OpenVoice/models/llama-3.2-Korean-Bllossom-3B",
    # model_kwargs={"torch_dtype": torch.bfloat16},
    device_map=device,
)

pipe.model.eval()

ckpt_converter = 'checkpoints_v2/converter'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')


# tts 모델 도커에서 다운 받은거 마운트 후에 snapshot -> checkpoint.pth, config.json을 형식에 맞게 모델입력으로 넣어줌 : 로컬에서 돌릴 수 있게됨
# from transformers import AutoModel
# model = AutoModel.from_pretrained("myshell-ai/MeloTTS-Korean", cache_dir="/app/OpenVoice/models/meloTTS-Korean")
# logging.info(model.config)
tts_path = "/root/.cache/huggingface/hub/models--myshell-ai--MeloTTS-Korean/snapshots/0207e5adfc90129a51b6b03d89be6d84360ed323/"

model = TTS(language='KR', config_path=tts_path + "config.json", ckpt_path=tts_path + "checkpoint.pth", device=device)

# model = TTS(language='KR', device=device)
speaker_ids = model.hps.data.spk2id


voice_man = "man.mp3"
man_reference_speaker = f'resources/{voice_man}'
target_se_man, audio_name_man = se_extractor.get_se(man_reference_speaker, tone_color_converter, vad=False)

voice_woman = "woman.mp3"
woman_reference_speaker = f'resources/{voice_woman}'
target_se_woman, audio_name_woman = se_extractor.get_se(woman_reference_speaker, tone_color_converter, vad=False)

speaker_key = 'kr'
source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)


def generation_text(keyword: str, speech: str, detail: None):
    start = time.time()

    keyword = keyword.replace(" ", "")

    # if detail:
    #     messages = [# Generate a sentence that uses the keyword "{keyword}".
    #         {"role": "system", "content": "너는 키워드 단어를 받아 해당 키워드에 대한 설명만을 한국어로 된 한 문장으로 간단하게 생성하는 훌룡한 조수야."},
    #         {"role": "user", "content": f"""{keyword}에 대해 모르는 사람에게 {keyword}를 소개하는 문장을 생성하세요.\n 
    #             detail은 {keyword}에 대한 설명입니다. 이를 참조하여 {keyword}에 대해 설명해주세요.\n
    #             문장에 {keyword}가 포함되어야 합니다. 또한, {keyword} 그대로를 출력하며 {keyword}를 제외한 어떠한 영어도 출력하지 마세요.\n
    #             문장 앞에 형용사를 넣으세요.\n
    #             {keyword}를 제외한 나머지 문장은 모두 한국어로 출력하세요.\n
    #             첫 번째 줄에는 반드시 한국어 문장을 작성해주세요.\n
    #             특수문자 없이 하나의 문장으로 작성하세요.\n
    #             10단어 이하의 문장을 만들어주세요.\n
    #             detail: {keyword}는 {detail}를 의미합니다.\n
    #             """}
    #     ]
    # else:
    #     messages = [
    #         {"role": "user", "content": f"""{keyword}에 대해 모르는 사람에게 {keyword}를 소개하는 문장을 생성하세요.\n 
    #             문장에 {keyword}가 포함되어야 합니다. 또한, {keyword} 그대로를 출력하며 {keyword}를 제외한 어떠한 영어도 출력하지 마세요.\n
    #             문장 앞에 형용사를 넣으세요.\n
    #             {keyword}를 제외한 나머지 문장은 모두 한국어로 출력하세요.\n
    #             첫 번째 줄에는 반드시 한국어 문장을 작성해주세요.\n
    #             특수문자 없이 하나의 문장으로 작성하세요.\n
    #             10단어 이하의 문장을 만들어주세요.\n
    #             """}
    #     ]

    if detail:
        messages = [{"role": "user", "content": f"""당신은 주어진 단어(Keyword)를 제외하고 모든 문장을 한국어로 출력하는 AI입니다.\n
아래 조건을 따르세요:\n
- 주어진 단어(Keyword)는 반드시 문장 중간 또는 마지막에 위치해야 합니다.\n
- 주어진 단어(Keyword) 외에는 모든 문장을 자연스러운 한국어로 작성하세요.\n
- 주어진 단어(Keyword)에 대한 추가 설명(Detail)을 참고하여 더욱 구체적인 문장을 작성하세요.\n
- 문장 안에 반드시 주어진 단어(Keyword)를 포함하세요.\n

예제:\n
[Keyword: DQN]\n
[Detail: 강화 학습 알고리즘 중 하나로, Q-learning을 딥러닝과 결합한 방식이다.]\n
출력: 강화 학습 알고리즘에는 [DQN] 알고리즘이 포함되어있다.\n
                     
[Keyword: TPS]\n
[Detail: 당사의 실시간 결제 승인 시스템을 의미]\n
출력: 실시간으로 결제 승인 여부를 판단하기 위해 [TPS]를 활용하여 신속한 결제 처리를 지원하고 있다.\n

[Keyword: AI]\n
[Detail: 다양한 산업 분야에서 AI에 대한 관심이 높아지며 기술을 도입하고 있다.]\n
출력: 산업 전반에서 [AI] 기술이 빠르게 도입되고 있다.\n
                     
[Keyword: BlueTiger]\n
[Detail: 당사에서 개발 중인 차세대 보안 솔루션의 코드명]\n
출력: 당사는 차세대 보안 시스템 [BlueTiger] 개발 프로젝트를 진행중이다.\n

이제 다음 키워드에 대해 문장을 생성하세요.\n

[Keyword: {keyword}]  
[Detail: {keyword}는 {detail}]  
출력:  
"""}]
        # messages = [# Generate a sentence that uses the keyword "{keyword}".
        #     {"role": "user", "content": f"""Generate a sentence that introduces {keyword} to someone who doesn't know about {keyword}.\n 
        #         detail is a description of {keyword}. Please explain {keyword} with reference to this.\n
        #         It must include {keyword} in sentence.\n
        #         Put an adjective in front of the sentence.\n
        #         Excluding {keyword}, print all remaining sentences in Korean.\n
        #         Be sure to create a Korean sentence in the first line.\n
        #         Create it as one sentence without special characters.\n
        #         Please create sentences of 10 words or less.\n
        #         detail: {keyword} means {detail}\n
        #         """}
        # ]
    else:
                messages = [{"role": "user", "content": f"""당신은 주어진 단어(Keyword)를 제외하고 모든 문장을 한국어로 출력하는 AI입니다.\n
아래 조건을 따르세요:\n
- 주어진 단어(Keyword)는 반드시 문장 중간 또는 마지막에 위치해야 합니다.\n
- 주어진 단어(Keyword) 외에는 모든 문장을 자연스러운 한국어로 작성하세요.\n
- 문장 안에 반드시 주어진 단어(Keyword)를 포함하세요.\n
- 반드시 주어진 단어(Keyword)를 포함한 일반적인 의미로 문장을 작성하세요.\n
                     
예제:\n
[Keyword: algorithm]\n
출력: 인공지능 모델을 개발할 때 가장 중요한 것은 최적의 [algorithm]을 설계하는 것입니다.\n
                     
[Keyword: sensor]\n
출력: 이 로봇은 온도와 습도를 감지할 수 있는 정밀한 [sensor]를 탑재하고 있다.\n

[Keyword: AI]\n
출력: 산업 전반에서 [AI] 기술이 빠르게 도입되고 있다.\n

이제 다음 키워드에 대해 문장을 생성하세요.\n

[Keyword: {keyword}]
출력:  
"""}]
        # messages = [
        #     {"role": "user", "content": f"""Generate a sentence that introduces {keyword} to someone who doesn't know about {keyword}.\n 
        #         It must include {keyword} in sentence.\n
        #         Put an adjective in front of the sentence.\n
        #         Excluding {keyword}, print all remaining sentences in Korean.\n
        #         Be sure to create a Korean sentence in the first line.\n
        #         Create it as one sentence without special characters.\n
        #         Please create sentences of 10 words or less.\n
        #         """}
        # ]

    output = pipe(messages, max_new_tokens=80, do_sample=True, temperature=0.01, top_p=0.98)
    logging.info(output[0]["generated_text"][-1]['content'])
    previous_text = output[0]["generated_text"][-1]['content'].split("assistant\n\n")[0].replace("\n\n", "\n")
    # previous_text = previous_text.replace("\n\n", "\n")
    # previous_text = previous_text.replace("\n \n", "\n")
    # previous_text = previous_text.replace("물론입니다.", "").replace("네,", "").strip()
    if len(previous_text.split("\n")) > 2:
        previous_text = previous_text.split("\n")[2]
    else:
        previous_text = previous_text.split("\n")[1]

    # result = ""
    # for text in previous_text.split('\n'):
    #     logging.info(text)
    #     if keyword in text.replace(" ", ""):
    #         result += text + "\n"
    #     else:
    #         num, _ = text.split(" ", 1)
    #         while True:
    #             message = [
    #                 {"role": "user", "content": f"""{keyword}를 포함한 새로운 문장을 생성해주세요.\n
    #                     반드시 키워드 {keyword} 그대로를 포함해야 합니다.\n
    #                     생성된 답변 이외의 텍스트는 출력하지 마세요.\n
    #                     문장은 간단하게 생성하고, 쉼표를 사용하지 않는 한 문장으로 생성하세요.\n
    #                     반드시 result를 참고하여 result엔 없는 새로운 문장을 생성해주세요.\n
    #                     result : {result}
    #                 """}
    #             ]
    #             output2 = pipe(message, max_new_tokens=35, do_sample=True, temperature=0.1, top_p=0.90, repetition_penalty=1.2, pad_token_id = pipe.tokenizer.eos_token_id)
    #             previous_text2 = output2[0]["generated_text"][-1]['content'].split("assistant\n\n")[0].split("assistant")[0] + "\n"

    #             if keyword in previous_text2.replace(" ", ""):
    #                 result += num + " " + previous_text2 + "\n"
    #                 break
    # logging.info(result)
    # result = result.replace("\n\n", "\n")
    
    
        # logging.info(previous_text)

    
    # if previous_text.split(".")[0].split(" ")[0] in ["here", "heres", "Here", "Heres"]:
    #     if previous_text.split(".")[0].split(":\n\n")[1].split(".")[0] == "1":
    #         previous_text = previous_text.split("1.")[1].split("\n")[0].strip()
    #     else:
    #         previous_text = previous_text.split(".")[0].split(":\n\n")[1]
    # else:
    #     if keyword.lower() in previous_text.split(".", 1)[0].lower():
    #         previous_text = previous_text.split(".", 1)[0]
    #     else:
    #         previous_text = previous_text.split(".", 1)[1]
    #         logging.info(previous_text + " asdf")

    #     if len(previous_text.split(".")) > 1:
    #         if keyword in previous_text.split(".")[0]:
    #             previous_text = previous_text.split(".")[0]
    #             logging.info(previous_text)
    #         else:
    #             previous_text = previous_text.split(".")[1]
    # if len(previous_text.split(":")) > 1:
    #     previous_text = previous_text.split(":", 1)[1].strip()
        
    previous_text = previous_text.replace("한국어: ", "")
    previous_text = previous_text.replace("Detail: ", "")
    previous_text = previous_text.replace("detail: ", "")
    previous_text = previous_text.replace("출력: ", "")
    previous_text = re.sub(r"[\\\[\]\"';]", "", previous_text) + "."
    previous_text = previous_text.replace("(", " ").replace(")", " ").replace("..", ".").strip()
    convert_text = previous_text.lower()
    
    convert_text = convert_text.replace(keyword.lower(), speech)
    
    end = time.time()
    complete_time = end - start
    logging.info(f"실행 시간: {complete_time:.2f}초")
    return {"previous_text": previous_text, "convert_text": convert_text}

def generation_texts(keyword: str, speech: str, detail: str):
    previous_texts = []
    convert_texts = []
    start = time.time()

    if detail:
        messages = [{"role": "user", "content": f"""당신은 주어진 단어(Keyword)를 제외하고 모든 문장을 한국어로 출력하는 AI입니다.\n
아래 조건을 따르세요:\n
- 주어진 단어(Keyword)는 반드시 문장 중간 또는 마지막에 위치해야 합니다.\n
- 주어진 단어(Keyword) 외에는 모든 문장을 자연스러운 한국어로 작성하세요.\n
- 주어진 단어(Keyword)에 대한 추가 설명(Detail)을 참고하여 더욱 구체적인 문장을 작성하세요.\n
- 문장 안에 반드시 주어진 단어(Keyword)를 포함하세요.\n
- 하나의 키워드에 대해 서로 다른 3개의 문장을 생성하세요.

예제:\n
[Keyword: DQN]\n
[Detail: 강화 학습 알고리즘 중 하나로, Q-learning을 딥러닝과 결합한 방식이다.]\n
출력:\n
1. 강화 학습에서 딥러닝 기법을 활용하는 대표적인 알고리즘이 [DQN]이다.\n
2. 로봇 제어와 같은 복잡한 환경에서도 [DQN]을 사용하면 효과적으로 학습할 수 있다.\n
3. 최근 연구에서는 기존 [DQN]을 개선한 변형 모델들이 제안되고 있다.\n
                     
[Keyword: TPS]\n
[Detail: 당사의 실시간 결제 승인 시스템을 의미]\n
출력:\n
1. 실시간 결제 승인 처리를 위해 [TPS]를 활용하고 있다.\n
2. 높은 트래픽에서도 안정적으로 작동하도록 [TPS] 시스템을 최적화하였다.\n
3. 사용자의 결제 요청이 들어오면 즉시 [TPS]에서 승인 여부를 판단한다.\n
                     
[Keyword: 운영체제]\n
[Detail: 컴퓨터 하드웨어를 관리하고 소프트웨어가 실행될 수 있는 환경을 제공하는 시스템 소프트웨어]\n
출력:\n
1. 다양한 프로그램을 실행하기 위해서는 안정적인 운영체제가 필요하다.\n
2. 최신 운영체제는 보안과 성능 최적화 기능이 강화되어 있다.\n
3. 새로운 소프트웨어를 설치하려면 운영체제와의 호환성을 확인해야 한다.\n

이제 다음 키워드에 대해 3개의 문장을 생성하세요.\n

[Keyword: {keyword}]\n
[Detail: {detail}]\n
출력:\n
1.  
2.  
3.  
"""}]
    else:
        messages = [{"role": "user", "content": f"""당신은 주어진 단어(Keyword)를 제외하고 모든 문장을 한국어로 출력하는 AI입니다.\n
아래 조건을 따르세요:\n
- 주어진 단어(Keyword)는 반드시 문장 중간 또는 마지막에 위치해야 합니다.\n
- 주어진 단어(Keyword) 외에는 모든 문장을 자연스러운 한국어로 작성하세요.\n
- 문장 안에 반드시 주어진 단어(Keyword)를 포함하세요.\n
- 반드시 주어진 단어(Keyword)를 포함한 일반적인 의미로 문장을 작성하세요.\n
- 하나의 키워드에 대해 서로 다른 3개의 문장을 생성하세요.
                     
예제:\n
[Keyword: algorithm]\n
출력:\n
1. 인공지능 모델을 개발할 때 가장 중요한 것은 최적의 [algorithm]을 설계하는 것입니다.\n
2. 새로운 데이터 분석 기법에서는 고급 [algorithm]이 필수적입니다.\n
3. 이 프로그램은 효율적인 [algorithm]을 사용하여 빠르게 계산을 수행합니다.\n
                     
[Keyword: sensor]\n
출력:\n
1. 이 로봇은 온도와 습도를 감지할 수 있는 정밀한 [sensor]를 탑재하고 있다.\n
2. 스마트폰에는 다양한 환경 변화를 감지하는 [sensor]가 내장되어 있다.\n
3. 자동차의 충돌 방지 시스템은 고성능 [sensor]를 활용하여 작동한다.\n
                         
[Keyword: 운영체제]\n
출력:\n
1. 다양한 프로그램을 실행하기 위해서는 안정적인 운영체제가 필요하다.\n
2. 최신 운영체제는 보안과 성능 최적화 기능이 강화되어 있다.\n
3. 새로운 소프트웨어를 설치하려면 운영체제와의 호환성을 확인해야 한다.\n

이제 다음 키워드에 대해 3개의 문장을 생성하세요.\n

[Keyword: {keyword}]\n
출력:\n
1.  
2.  
3.  
"""}]
    # if detail:
    #     messages = [{"role": "user", "content": f"""상세설명을 참고하여 {keyword}를 설명하는 문장 세개를 사전 형식으로 생성하세요.\n
    #         열가지 문장은 모두 한국어로 생성하고 각각의 문장은 개별적인 문장이며, 서로 연관없게 작성하세요.\n
    #         1번부터 3번까지 모두 반드시 "{keyword}" 단어 그대로 포함되어야 하며, 열 단어 이하의 짧은 문장으로 생성하세요.\n
    #         "{keyword}"를 문장의 중간 또는 맨 뒤에 배치하세요.\n
    #         상세설명 : {detail}\n\n
    #         출력 형식은 아래와 같습니다.\n
    #         1. {keyword}를 포함한 간단한 문장 \n
    #         2. {keyword}를 포함한 간단한 문장 \n
    #         3. {keyword}를 포함한 간단한 문장\n
    #         """}]
    # else:
    #     messages = [{"role": "user", "content": f"""{keyword}를 설명하는 문장 세개를 사전 형식으로 생성하세요.\n
    #         열가지 문장은 모두 한국어로 생성하고 각각의 문장은 개별적인 문장이며, 서로 연관없게 작성하세요.\n
    #         1번부터 3번까지 모두 반드시 "{keyword}" 단어 그대로 포함되어야 하며, 열 단어 이하의 짧은 문장으로 생성하세요.\n
    #         "{keyword}"를 문장의 중간 또는 맨 뒤에 배치하세요\n
    #         출력 형식은 아래와 같습니다.\n
    #         1. {keyword}를 포함한 간단한 문장 \n
    #         2. {keyword}를 포함한 간단한 문장 \n
    #         3. {keyword}를 포함한 간단한 문장\n
    #         """}]

    output = pipe(messages, max_new_tokens=250, do_sample=True, temperature=0.01, top_p=0.98)
    logging.info(output[0]["generated_text"][-1]['content'].split("assistant\n\n")[0])
    previous_text = output[0]["generated_text"][-1]['content'].split("assistant\n\n")[0]
    previous_text = previous_text.replace("\n\n", "\n").split("\n",1)[1].strip()
    if previous_text.split(" ", 1)[0] == "[Detail:":
        logging.info(previous_text)
        previous_text = previous_text.split("\n", 1)[1]

    previous_text = previous_text.replace("출력: ", "")
    previous_text = previous_text.replace("Detail: ", "")
    previous_text = re.sub(r"\b[1-3]\.\s*", "", previous_text)
    previous_text = re.sub(r"[\\\[\]\"';]", "", previous_text) + "."
    previous_text = previous_text.replace("(", " ").replace(")", " ").replace("..", ".").strip()
    convert_text = previous_text.lower().replace(keyword.lower(), speech)
    
    previous_texts.append(previous_text)
    convert_texts.append(convert_text)
    end = time.time()
    complete_time = end - start
    logging.info(f"실행 시간: {complete_time:.2f}초")
    return {"previous_texts": previous_texts, "convert_texts": convert_texts}


def save_texts(keyword: str, speech: str, detail: str):
    previous_texts = []
    convert_texts = []
    start = time.time()

    if detail:          
        messages = [{"role": "user", "content": f"""당신은 주어진 단어(Keyword)를 제외하고 모든 문장을 한국어로 출력하는 AI입니다.\n
아래 조건을 따르세요:\n
- 주어진 단어(Keyword)는 반드시 문장 중간 또는 마지막에 위치해야 합니다.\n
- 주어진 단어(Keyword) 외에는 모든 문장을 자연스러운 한국어로 작성하세요.\n
- 주어진 단어(Keyword)에 대한 추가 설명(Detail)을 참고하여 더욱 구체적인 문장을 작성하세요.\n
- 문장 안에 반드시 주어진 단어(Keyword)를 포함하세요.\n

예제:\n
[Keyword: DQN]\n
[Detail: 강화 학습 알고리즘 중 하나로, Q-learning을 딥러닝과 결합한 방식이다.]\n
출력:\n
1. 강화 학습에서 딥러닝 기법을 활용하는 대표적인 알고리즘이 [DQN]이다.\n
2. 로봇 제어와 같은 복잡한 환경에서도 [DQN]을 사용하면 효과적으로 학습할 수 있다.\n
3. 최근 연구에서는 기존 [DQN]을 개선한 변형 모델들이 제안되고 있다.\n
                     
[Keyword: TPS]\n
[Detail: 당사의 실시간 결제 승인 시스템을 의미]\n
출력:\n
1. 실시간 결제 승인 처리를 위해 [TPS]를 활용하고 있다.\n
2. 높은 트래픽에서도 안정적으로 작동하도록 [TPS] 시스템을 최적화하였다.\n
3. 사용자의 결제 요청이 들어오면 즉시 [TPS]에서 승인 여부를 판단한다.\n
                     
[Keyword: 운영체제]\n
[Detail: 컴퓨터 하드웨어를 관리하고 소프트웨어가 실행될 수 있는 환경을 제공하는 시스템 소프트웨어]\n
출력:\n
1. 다양한 프로그램을 실행하기 위해서는 안정적인 운영체제가 필요하다.\n
2. 최신 운영체제는 보안과 성능 최적화 기능이 강화되어 있다.\n
3. 새로운 소프트웨어를 설치하려면 운영체제와의 호환성을 확인해야 한다.\n

이제 다음 키워드에 대해 3개의 문장을 생성하세요.\n

[Keyword: {keyword}]\n
[Detail: {detail}]\n
출력:\n
1.  
2.  
3.  
"""}]
    else:
        messages = [{"role": "user", "content": f"""당신은 주어진 단어(Keyword)를 제외하고 모든 문장을 한국어로 출력하는 AI입니다.\n
아래 조건을 따르세요:\n
- 주어진 단어(Keyword)는 반드시 문장 중간 또는 마지막에 위치해야 합니다.\n
- 주어진 단어(Keyword) 외에는 모든 문장을 자연스러운 한국어로 작성하세요.\n
- 문장 안에 반드시 주어진 단어(Keyword)를 포함하세요.\n
- 반드시 주어진 단어(Keyword)를 포함한 일반적인 의미로 문장을 작성하세요.\n
                     
예제:\n
[Keyword: algorithm]\n
출력:\n
1. 인공지능 모델을 개발할 때 가장 중요한 것은 최적의 [algorithm]을 설계하는 것입니다.\n
2. 새로운 데이터 분석 기법에서는 고급 [algorithm]이 필수적입니다.\n
3. 이 프로그램은 효율적인 [algorithm]을 사용하여 빠르게 계산을 수행합니다.\n
                     
[Keyword: sensor]\n
출력:\n
1. 이 로봇은 온도와 습도를 감지할 수 있는 정밀한 [sensor]를 탑재하고 있다.\n
2. 스마트폰에는 다양한 환경 변화를 감지하는 [sensor]가 내장되어 있다.\n
3. 자동차의 충돌 방지 시스템은 고성능 [sensor]를 활용하여 작동한다.\n
                         
[Keyword: 운영체제]\n
출력:\n
1. 다양한 프로그램을 실행하기 위해서는 안정적인 운영체제가 필요하다.\n
2. 최신 운영체제는 보안과 성능 최적화 기능이 강화되어 있다.\n
3. 새로운 소프트웨어를 설치하려면 운영체제와의 호환성을 확인해야 한다.\n

이제 다음 키워드에 대해 3개의 문장을 생성하세요.\n

[Keyword: {keyword}]\n
출력:\n
"""}]
# 1.  
# 2.  
# 3.  
    # if detail:
    #     messages = [{"role": "user", "content": f"""상세설명을 참고하여 {keyword}를 설명하는 문장 세개를 사전 형식으로 생성하세요.\n
    #         열가지 문장은 모두 한국어로 생성하고 각각의 문장은 개별적인 문장이며, 서로 연관없게 작성하세요.\n
    #         1번부터 3번까지 모두 반드시 "{keyword}" 단어 그대로 포함되어야 하며, 열 단어 이하의 짧은 문장으로 생성하세요.\n
    #         "{keyword}"를 문장의 중간 또는 맨 뒤에 배치하세요.\n
    #         상세설명 : {detail}\n\n
    #         출력 형식은 아래와 같습니다.\n
    #         1. {keyword}를 포함한 간단한 문장 \n
    #         2. {keyword}를 포함한 간단한 문장 \n
    #         3. {keyword}를 포함한 간단한 문장\n
    #         """}]
    # else:
    #     messages = [{"role": "user", "content": f"""{keyword}를 설명하는 문장 세개를 사전 형식으로 생성하세요.\n
    #         열가지 문장은 모두 한국어로 생성하고 각각의 문장은 개별적인 문장이며, 서로 연관없게 작성하세요.\n
    #         1번부터 3번까지 모두 반드시 "{keyword}" 단어 그대로 포함되어야 하며, 열 단어 이하의 짧은 문장으로 생성하세요.\n
    #         "{keyword}"를 문장의 중간 또는 맨 뒤에 배치하세요\n
    #         출력 형식은 아래와 같습니다.\n
    #         1. {keyword}를 포함한 간단한 문장 \n
    #         2. {keyword}를 포함한 간단한 문장 \n
    #         3. {keyword}를 포함한 간단한 문장\n
    #         """}]

    output = pipe(messages, max_new_tokens=100, do_sample=True, temperature=0.01, top_p=0.98)
    logging.info(output[0]["generated_text"][-1]['content'])#.split("assistant\n\n")[0]
    previous_text = output[0]["generated_text"][-1]['content'].split("assistant\n\n")[0]
    previous_text = previous_text.replace("\n\n", "\n").strip()
    if previous_text.split(" ", 1)[0] == "[Detail:":
        logging.info(previous_text)
        previous_text = previous_text.split("\n", 1)[1]

    previous_text = previous_text.replace("출력: ", "")
    previous_text = previous_text.replace("Detail: ", "")
    previous_text = re.sub(r"\b[1-3]\.\s*", "", previous_text)
    previous_text = re.sub(r"[\\\[\]\"';]", "", previous_text) + "."
    previous_text = previous_text.replace("(", " ").replace(")", " ").replace("..", ".").strip()
    convert_text = previous_text.lower().replace(keyword.lower(), speech)
    
    previous_texts.extend(previous_text.split("\n"))
    convert_texts.extend(convert_text.split("\n"))
    end = time.time()
    complete_time = end - start
    logging.info(f"실행 시간: {complete_time:.2f}초")

    for i in range(len(convert_texts)):
        with open(f"""/app/OpenVoice/save_text_data/kw_{keyword}_{i+1}.txt""", "w", encoding="utf-8") as f:
            f.write(convert_texts[i])
            print(convert_texts[i], )

    return {"previous_texts": previous_texts, "convert_texts": convert_texts}


def save_tts(voice: str, sentence: str):
    start = time.time()
    if voice=="man":
        target_se = target_se_man
        audio_name = audio_name_man
    else:
        target_se = target_se_woman
        audio_name = audio_name_woman

    speed = 1.3

    generated_text = ".." + sentence

    output_path = f"/app/OpenVoice/outputs_v2"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = f"{output_path}/{os.path.splitext(voice)[0]}.wav"

    # model.tts_to_file("dkss", speaker_ids['KR'], output_path, speed=speed)
    model.tts_to_file(generated_text, speaker_ids['KR'], output_path, speed=speed)

    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=output_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=output_path,
        message=encode_message)
    end = time.time()
    complete_time = end - start
    logging.info(f"실행 시간: {complete_time:.2f}초")
    return FileResponse(output_path, media_type='audio/wav', filename=f'{voice.split(".mp3")[0]}.wav')


def save_tts_ex(voice: str):
    file_list = [txt.split(".")[0] for txt in os.listdir("/app/OpenVoice/save_text_data/") if txt.endswith(".txt")]

    start = time.time()
    if voice=="man":
        target_se = target_se_man
        audio_name = audio_name_man
    else:
        target_se = target_se_woman
        audio_name = audio_name_woman

    speed = 1.3

    print(len(file_list))
    for txt_file in file_list:
        print(txt_file)
        file_path = os.path.join("/app/OpenVoice/save_text_data/", txt_file)
        with open(file_path + ".txt", 'r', encoding='utf-8') as file:
            sentence = file.read()
        generated_text = ".." + sentence

        output_path = f"/app/OpenVoice/save_tts_data"
        output_path = f"{output_path}/{txt_file}_{voice[0]}.wav"

        # model.tts_to_file("dkss", speaker_ids['KR'], output_path, speed=speed)
        model.tts_to_file(generated_text, speaker_ids['KR'], output_path, speed=speed)

        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=output_path, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=output_path,
            message=encode_message)
    end = time.time()
    complete_time = end - start
    logging.info(f"실행 시간: {complete_time:.2f}초")
    return FileResponse(output_path, media_type='audio/wav', filename=f"{txt_file}_{voice[0]}.wav")