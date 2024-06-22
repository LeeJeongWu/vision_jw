import asyncio
import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import sys
import time

# BitsAndBytesConfig 설정
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["mm_projector", "vision_model"],
)

# 모델 및 토크나이저 로드
model_id = "openbmb/MiniCPM-Llama3-V-2_5-int4"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True, 
    # torch_dtype=torch.float16,
    # quantization_config=bnb_cfg,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

PROMPT = """What kind of activities and emotions are in this video?. Answer following JSON format,
{"emotions": emotions, "activities": actions}
### Reply JSON but nothing else.###"""
# PROMPT= """Tell us what the person in the frame is feeling and doing."""

# 비동기적으로 모델 추론을 수행하는 함수
async def infer(frames, prompt):
    contents = [prompt]
    contents.extend(frames)
    res = model.chat(
        msgs=[
            {'role': 'user', 'content': contents}
        ],
        context=None,
        image=None,
        tokenizer=tokenizer,
        do_sample=True,
        max_new_tokens=32,
    )
    return res

async def analyze_video(file_path):
    cap = cv2.VideoCapture(file_path)
    prev_time = time.time_ns()
    frames = []
    results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur_time = time.time_ns()
        frameS = cv2.resize(frame, (720, 480))
        frame_pil = Image.fromarray(frameS)
        if cur_time - prev_time > 100_000_000:
            frames.append(frame_pil)
            prev_time = cur_time
        if len(frames) > 3:
            frames = frames[-3:]
        # 비동기 추론 호출
        response = await infer(frames, PROMPT)
        results.append(response)

    cap.release()
    return results

if __name__ == '__main__':
    video_file = sys.argv[1]
    result = asyncio.run(analyze_video(video_file))
    print(result)