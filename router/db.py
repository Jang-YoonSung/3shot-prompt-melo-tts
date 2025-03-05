from fastapi import APIRouter, Query, Body

from service import db_service

import os

import logging


router = APIRouter()

@router.post("/generation-text")
async def generation_text(keyword: str = Body(...),
                          speech: str = Body(...),
                          detail: str = Body(None)):
    return db_service.generation_text(keyword, speech, detail)

@router.post("/generation-texts")
async def generation_texts(keyword: str = Body(...),
                          speech: str = Body(...),
                          detail: str = Body(None)):
    return db_service.generation_texts(keyword, speech, detail)

@router.post("/save-tts")
async def save_tts(voice: str = Body("man or woman or y_man or y_woman", enum=[mp3 for mp3 in os.listdir("/app/OpenVoice/resources") if mp3.endswith(".mp3")]),
                    sentence: str = Body(...)):
    return db_service.save_tts(voice, sentence)

# @router.post("/save_tts_ex")
# async def save_tts_ex(voice: str = Body("man or woman or young", enum=[mp3 for mp3 in os.listdir("/app/OpenVoice/resources") if mp3.endswith(".mp3")]),
#                       sentence: str = Body(...)):
#     return db_service.save_tts_ex(voice, sentence)