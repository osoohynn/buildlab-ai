import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from transformers import pipeline
import uvicorn

# Sumy 라이브러리 임포트
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 번역 함수 (LibreTranslate API 사용)
def translate_text(text: str, src="ko", dest="en") -> str:
    try:
        url = "https://api.mymemory.translated.net/get"
        params = {"q": text, "langpair": f"{src}|{dest}"}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            result = response.json()
            translated_text = result["responseData"]["translatedText"]
            return translated_text
        else:
            logger.error(f"Translation API error: {response.status_code}")
            return "Translation API error"
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return f"Translation error: {str(e)}"

# 영어 상태에서 추출 요약을 수행하는 함수 (Sumy의 LexRank 사용)
def extract_key_content(text: str, num_sentences: int = 1) -> str:
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary_sentences = summarizer(parser.document, num_sentences)
        summary = " ".join(str(sentence) for sentence in summary_sentences)
        return summary
    except Exception as e:
        logger.error("Extract key content error: %s", e)
        return "Key content extraction error"

# 감정 분석: Zero-shot classification (facebook/bart-large-mnli)
try:
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    candidate_labels = [
        "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise",
        "excitement", "contentment", "anxiety", "boredom"
    ]
    logger.info("Zero-shot emotion analysis model initialized successfully")
except Exception as e:
    zero_shot_classifier = None
    candidate_labels = []
    logger.error("Zero-shot emotion analysis model initialization failed: %s", e)

# Pydantic 모델 정의
class MessageInput(BaseModel):
    message: str

@app.post("/analyze")
def analyze_message(input: MessageInput):
    results = {
        "translatedText": "",
        "happen": "",
        "emotions": []
    }
    try:
        # 1. 한국어 입력을 영어로 번역 (KO -> EN)
        english_text = translate_text(input.message, src="ko", dest="en")
        if english_text and not english_text.startswith("Translation"):
            results["translatedText"] = english_text
            logger.info("Translation (KO->EN) successful")
        else:
            results["translatedText"] = "Translation failed"
            logger.error("Translation failed: %s", english_text)
        
        # 2. 영어 텍스트에서 핵심 내용 추출 (추출 요약)
        if english_text and english_text != "Translation failed":
            key_content = extract_key_content(english_text, num_sentences=1)
            logger.info("Key content extraction successful: %s", key_content)
        else:
            key_content = "Summarization skipped"
            logger.info("No valid English text, summarization skipped")
        
        # 3. 영어 요약 결과를 한국어로 번역 (EN -> KO)
        korean_summary = translate_text(key_content, src="en", dest="ko")
        results["happen"] = korean_summary
        logger.info("Translation (EN->KO) of summary successful: %s", korean_summary)
        
        # 4. 감정 분석: 영어 번역문을 기반으로 Zero-shot 분류 수행
        if zero_shot_classifier and results["translatedText"] != "Translation failed":
            classification = zero_shot_classifier(results["translatedText"], candidate_labels)
            emotions = []
            for label, score in zip(classification["labels"], classification["scores"]):
                emotions.append({"label": label, "score": score})
            results["emotions"] = emotions
            logger.info("Emotion analysis successful")
        else:
            results["emotions"] = []
            logger.info("Emotion analysis skipped due to translation failure or classifier init failure")
        
        return results
    except Exception as e:
        logger.error("Overall analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "services": {
            "extractive_summarization": True,  # Sumy 사용 여부
            "zero_shot_emotion_analysis": zero_shot_classifier is not None
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
