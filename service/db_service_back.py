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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    "text-generation",
    # model="meta-llama/Llama-3.2-1B-Instruct",
    # model="meta-llama/Llama-3.2-3B-Instruct",
    model="/app/OpenVoice/models/llama-3.2-Korean-Bllossom-3B",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=device,
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

logging.basicConfig(level=logging.INFO)

def generation_text(keyword: str, speech: str, detail: None):
    start = time.time()

    if detail:
        messages = [# Generate a sentence that uses the keyword "{keyword}".
            {"role": "user", "content": f"""Generate a sentence that introduces {keyword} to someone who doesn't know about {keyword}.\n 
                detail is a description of {keyword}. Please explain {keyword} with reference to this.\n
                It must include {keyword} in sentence.\n
                Excluding {keyword}, print all remaining sentences in Korean.\n
                Be sure to create a Korean sentence in the first line.\n
                Create it as one sentence without special characters.\n
                Please create sentences of 10 words or less.\n
                detail: {keyword} means {detail}
                """}
        ]
    else:
        messages = [
            {"role": "user", "content": f"""Generate a sentence that introduces {keyword} to someone who doesn't know about {keyword}.\n 
                It must include {keyword} in sentence.\n
                Excluding {keyword}, print all remaining sentences in Korean.\n
                Be sure to create a Korean sentence in the first line.\n
                Create it as one sentence without special characters.\n
                Please create sentences of 10 words or less.\n
                """}
        ]

    # output = pipe(messages, max_new_tokens=50, do_sample=True, temperature=0.01, top_p=0.98)
    # previous_text = output[0]["generated_text"][-1]['content'].split("assistant\n\n")[0].split(".")[0]
    # previous_text = re.sub(r"\([^)]*\)", "", previous_text)
    # previous_text = re.sub(r"[\\\"';]", "", previous_text)
    # convert_text = previous_text.lower()
    
    # output = pipe(messages, max_new_tokens=50, do_sample=True, temperature=0.01, top_p=0.98)
    # previous_text = output[0]["generated_text"][-1]['content'].split("assistant\n\n")[0]
    # previous_text = previous_text.replace("물론입니다.", "").replace("네,", "").strip()
    # logging.info(previous_text)

    # if previous_text.split(".")[0].split(" ")[0] in ["here", "heres", "Here", "Heres"]:
    #     if previous_text.split(".")[0].split(":\n\n")[1].split(".")[0] == "1":
    #         previous_text = previous_text.split("1.")[1].split("\n")[0].strip()
    #     else:
    #         previous_text = previous_text.split(".")[0].split(":\n\n")[1]
    # else:
    #     previous_text = previous_text.split(".")[0]
    #     logging.info(previous_text)
    # if ":" in previous_text:
    #     previous_text = previous_text.split("\n\n")[-1]
    # previous_text = re.sub(r"\([^)]*\)-", "", previous_text)
    # previous_text = previous_text.replace("한국어: ", "")
    # previous_text = re.sub(r"[\\\[\]\"';]", "", previous_text) + "."
    # convert_text = previous_text.lower()

    
    # messages = [
    #     {"role": "user", "content": f"""{keyword}를 설명하는 문장 열개를 사전 형식으로 생성하세요.\n
    #         열가지 문장은 모두 한국어로 생성하고 각각의 문장은 개별적인 문장이며, 서로 연관없게 작성하세요.\n
    #         1번부터 10번까지 모두 반드시 {keyword} 단어 그대로 포함되어야 하며, 열 단어 이하의 짧은 문장으로 생성하세요.\n\n
    #         출력 형식은 아래와 같습니다.\n
    #         1. {keyword} ... \n
    #         2. {keyword} ... \n
    #         ...
    #         10. {keyword} ...\n
    #         """}
    # ]

    # output = pipe(messages, max_new_tokens=50, do_sample=True, temperature=0.01, top_p=0.98)
    # previous_text = output[0]["generated_text"][-1]['content'].split("assistant\n\n")[0].split(".")[0]
    # previous_text = re.sub(r"\([^)]*\)", "", previous_text)
    # previous_text = re.sub(r"[\\\"';]", "", previous_text)
    # convert_text = previous_text.lower()
    
    # output = pipe(messages, max_new_tokens=50, do_sample=True, temperature=0.01, top_p=0.98)
    # previous_text = output[0]["generated_text"][-1]['content'].split("assistant\n\n")[0]
    # previous_text = previous_text.replace("물론입니다.", "").replace("네,", "").strip()
    # logging.info(previous_text)

    # if previous_text.split(".")[0].split(" ")[0] in ["here", "heres", "Here", "Heres"]:
    #     if previous_text.split(".")[0].split(":\n\n")[1].split(".")[0] == "1":
    #         previous_text = previous_text.split("1.")[1].split("\n")[0].strip()
    #     else:
    #         previous_text = previous_text.split(".")[0].split(":\n\n")[1]
    # else:
    #     previous_text = previous_text.split(".")[0]
    #     logging.info(previous_text)
    # if ":" in previous_text:
    #     previous_text = previous_text.split("\n\n")[-1]
    # previous_text = re.sub(r"\([^)]*\)-", "", previous_text)
    # previous_text = previous_text.replace("한국어: ", "")
    # previous_text = re.sub(r"[\\\[\]\"';]", "", previous_text) + "."
    # convert_text = previous_text.lower()

    output = pipe(messages, max_new_tokens=50, do_sample=True, temperature=0.1, top_p=0.90)
    previous_text = output[0]["generated_text"][-1]['content'].split("assistant\n\n")[0]
    previous_text = previous_text.replace("물론입니다.", "").replace("네,", "").strip()
    logging.info(previous_text)
    
    if previous_text.split(".")[0].split(" ")[0] in ["here", "heres", "Here", "Heres"]:
        if previous_text.split(".")[0].split(":\n\n")[1].split(".")[0] == "1":
            previous_text = previous_text.split("1.")[1].split("\n")[0].strip()
        else:
            previous_text = previous_text.split(".")[0].split(":\n\n")[1]
    else:
        if keyword.lower() in previous_text.split(".", 1)[0].lower():
            previous_text = previous_text.split(".", 1)[0]
            logging.info(previous_text)
        else:
            previous_text = previous_text.split(".", 1)[1]
            # logging.info(previous_text)

        if len(previous_text.split(".")) > 1:
            if keyword in previous_text.split(".")[0]:
                previous_text = previous_text.split(".")[0]
            else:
                previous_text = previous_text.split(".")[1]
        
    previous_text = previous_text.replace("한국어: ", "")
    previous_text = previous_text.replace("Detail: ", "")
    previous_text = previous_text.replace("detail: ", "")
    previous_text = re.sub(r"[\\\[\]\"';]", "", previous_text) + "."
    previous_text = previous_text.replace("(", " ").replace(")", " ").strip()
    logging.info(previous_text)
    convert_text = previous_text.lower()
    
    convert_text = convert_text.replace(keyword.lower(), speech)
    end = time.time()
    complete_time = end - start
    # logging.info(f"실행 시간: {complete_time:.2f}초")
    return {"previous_text": previous_text, "convert_text": convert_text}

# def generation_texts(keywords: list, speeches: list, detail: list):
#     result = []
#     if len(keywords) != len(speeches):
#         raise HTTPException(status_code=404, detail="keywords와 speeches의 길이가 다릅니다.")
#     for keyword, speech in zip(keywords, speeches):
#         messages = [
#         {"role": "user", "content": f""""Generate a sentence that uses the keyword {keyword}.\n
#             Only include Korean text, and ensure that no characters from other languages appear.\n
#             Provide a single sentence in Korean without any special characters
#             """}
#         ]

#         output = pipe(messages, max_new_tokens=50, do_sample=True, temperature=0.05, top_p=0.98)
#         output_text = output[0]["generated_text"][-1]['content'].split("assistant\n\n")[0]
#         if speech:
#             output_text = output_text.lower().replace(keyword, speech)
#         output_text = re.sub(r"[\"/]", "", output_text)
        
#         result.append(output_text)
#     return result


def save_tts(voice: str, sentence: str):
    start = time.time()
    # if voice != "man" and voice != "woman":
    #     raise HTTPException(status_code=404, detail="man 또는 woman을 입력해주세요.")
    # voice = voice + ".mp3"
    # reference_speaker = f'resources/{voice}'
    # logging.info(tone_color_converter)
    # target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)
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

    model.tts_to_file(generated_text, speaker_ids['KR'], output_path, speed=speed)

    speaker_key = 'kr'
    source_se = torch.load(f'checkpoints_v2/base_speakers/ses/{speaker_key}.pth', map_location=device)
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