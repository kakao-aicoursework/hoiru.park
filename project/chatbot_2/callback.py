from dto import ChatbotRequest
import aiohttp
import time
import logging
import openai
import os
import requests
from chatbot_2 import generate_sync_bot

# 환경 변수 처리 필요!
openai.api_key = os.environ['API_KEY']
SYSTEM_MSG = "당신은 카카오 서비스 제공자입니다."
logger = logging.getLogger("Callback")


def callback_handler(request: ChatbotRequest) -> dict:
    # ===================== start =================================
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": request.userRequest.utterance},
        ],
        temperature=0,
    )
    # focus
    output_text = generate_sync_bot(request.userRequest.utterance)
    print(output_text)

    # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
                    }
                }
            ]
        }
    }
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    time.sleep(0.1)

    url = request.userRequest.callbackUrl

    if url:
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(url=url, json=payload, ssl=False) as resp:
        #         await resp.json()
        requests.post(url, json=payload)
