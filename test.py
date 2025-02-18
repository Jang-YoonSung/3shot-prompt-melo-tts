import logging
import os

from melo.api import TTS

# open voice 관련 라이브러리
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# huggingface 관련 모델 라이브러리
from transformers import pipeline
import time
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

ckpt_converter = 'checkpoints_v2/converter'

tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

tts_path = "/root/.cache/huggingface/hub/models--myshell-ai--MeloTTS-Korean/snapshots/0207e5adfc90129a51b6b03d89be6d84360ed323/"
model = TTS(language='KR', use_hf=False, config_path=tts_path + "config.json", ckpt_path=tts_path + "checkpoint.pth", device=device)
speaker_ids = model.hps.data.spk2id

voice_man = "man.mp3"
man_reference_speaker = f'resources/{voice_man}'
target_se_man, audio_name_man = se_extractor.get_se(man_reference_speaker, tone_color_converter, vad=False)

logging.basicConfig(level=logging.INFO)

start = time.time()
# if voice != "man" and voice != "woman":
#     raise HTTPException(status_code=404, detail="man 또는 woman을 입력해주세요.")
# voice = voice + ".mp3"
# reference_speaker = f'resources/{voice}'
# logging.info(tone_color_converter)
# target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)
voice = "man"
target_se = target_se_man
audio_name = audio_name_man

speed = 1.3

sentence = "안녕하세요. 티티에스 테스트 입니다."

generated_text = ".." + sentence

output_path = f"/app/OpenVoice/outputs_v2"

if not os.path.exists(output_path):
    os.makedirs(output_path)

output_path = f"{output_path}/{os.path.splitext(voice)[0]}.wav"

# model.tts_to_file("dkss", speaker_ids['KR'], output_path, speed=speed)
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

print(complete_time, "폐쇄망 테스트 통과")