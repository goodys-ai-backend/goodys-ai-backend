import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from groq import Groq
import google.generativeai as genai

app = FastAPI()

# API 키 설정 (Vercel 환경변수에서 불러옴)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 클라이언트 초기화
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

class ModelRequest(BaseModel):
    model_name: str

@app.post("/compare")
async def compare_models(request: ModelRequest):
    model_name = request.model_name

    # 1. Groq에게 해당 모델의 장점/속도 물어보기
    try:
        groq_resp = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": f"{model_name} 모델의 성능과 추론 속도 특이점을 짧게 알려줘."}],
            model="llama-3.1-70b-versatile",
        )
        groq_info = groq_resp.choices[0].message.content
    except:
        groq_info = "Groq 정보를 가져오지 못했습니다."

    # 2. Hugging Face에서 모델 상세 사양(Tags, 파라미터 등) 검색
    try:
        hf_url = f"https://huggingface.co/api/models/{model_name}"
        hf_resp = requests.get(hf_url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
        hf_data = hf_resp.json() if hf_resp.status_code == 200 else "정보 없음"
    except:
        hf_data = "HF 정보를 가져오지 못했습니다."

    # 3. Gemini에게 최종 비교 및 권장안 요청
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        사용자가 궁금해하는 모델: {model_name}
        
        [Groq의 의견]: {groq_info}
        [Hugging Face 데이터]: {hf_data}
        
        위 정보를 바탕으로 이 모델을 써야 할지 말지, 어떤 용도에 최적인지 한국어로 요약해서 전문가 의견을 내놔.
        """
        gemini_resp = gemini_model.generate_content(prompt)
        final_answer = gemini_resp.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "model": model_name,
        "analysis": final_answer
    }
