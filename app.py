from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import asyncio
import base64
from io import BytesIO
import os

app = Flask(__name__)

# CUDA_LAUNCH_BLOCKING 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

PROMPT = """What kind of activities and emotions are in this video? Describe in one sentence."""

async def infer(frames, prompt):
    contents = [prompt]
    contents.extend(frames)
    try:
        res = model.chat(
            msgs=[
                {'role': 'user', 'content': contents}
            ],
            context=None,
            image=None,
            tokenizer=tokenizer,
            do_sample=False,
            max_new_tokens=32,
        )
        print(f"Infer response: {res}")  # 응답 구조 확인을 위한 디버깅 출력
        return res
    except Exception as e:
        print(f"Error during inference: {e}")
        return {'choices': [{'text': 'Error during inference.'}]}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    frames = request.json.get('frames')
    if not frames:
        return jsonify({'error': 'No frames provided'}), 400
    
    try:
        frames = [Image.open(BytesIO(base64.b64decode(frame.split(',')[1]))) for frame in frames]
        print(f"Frames: {frames}")  # 디버깅을 위한 프레임 출력
        
        # 첫 번째 추론: 행동과 감정 인식 문장 생성
        response = asyncio.run(infer(frames, PROMPT))
        response_text = response['choices'][0]['text'] if isinstance(response, dict) else response
        
        # 두 번째 추론: 인식 문장에 대한 응답 문장 생성
        response_to_response = asyncio.run(infer([response_text], "### Reply to the above text. ###"))
        response_to_response_text = response_to_response['choices'][0]['text'] if isinstance(response_to_response, dict) else response_to_response
        
        return jsonify({
            'response': response_text,
            'response_to_response': response_to_response_text
        })
    except Exception as e:
        print(f"Error during analyze: {e}")
        return jsonify({'error': 'Error during analysis.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)