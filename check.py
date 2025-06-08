# check_models.py
import os
import requests
from dotenv import load_dotenv

def check_google_ai_models():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ GOOGLE_API_KEY가 설정되지 않았습니다.")
        return
    
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        headers = {"x-goog-api-key": api_key}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            models = response.json()
            print("🔍 사용 가능한 Google AI 모델들:")
            print("-" * 50)
            
            for model in models.get("models", []):
                model_name = model.get("name", "")
                display_name = model.get("displayName", "")
                supported_methods = model.get("supportedGenerationMethods", [])
                
                if "generateContent" in supported_methods:
                    clean_name = model_name.replace("models/", "")
                    print(f"✅ {clean_name}")
                    if display_name:
                        print(f"   설명: {display_name}")
                    print()
        else:
            print(f"❌ API 요청 실패: {response.status_code}")
            print(f"응답: {response.text}")
            
    except Exception as e:
        print(f"❌ 모델 확인 중 오류: {e}")

if __name__ == "__main__":
    check_google_ai_models()