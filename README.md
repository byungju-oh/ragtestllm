# 🔍 RAG Optimization Comparison System

FastAPI + LangChain + Google AI Studio를 활용한 **RAG(Retrieval-Augmented Generation) 최적화 방법 비교 시스템**입니다. 다양한 RAG 기법들의 성능을 실시간으로 비교하고 분석할 수 있습니다.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?style=flat&logo=react&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=flat&logo=chainlink&logoColor=white)
![Google AI](https://img.shields.io/badge/Google_AI-4285F4?style=flat&logo=google&logoColor=white)

## ✨ 주요 기능

### 🎯 RAG 최적화 방법 비교
- **Basic RAG**: 기본적인 벡터 검색 + LLM 생성
- **Multi-Query RAG**: 다중 쿼리로 검색 다양성 확보
- **Ensemble Retrieval**: 벡터 검색 + BM25 키워드 검색 결합
- **Contextual Compression**: 관련 정보만 선별하여 컨텍스트 최적화
- **Hierarchical Chunking**: 계층적 문서 분할로 구조화
- **Semantic Chunking**: 의미 기반 문서 분할

### 📊 성능 분석 및 시각화
- 실시간 성능 비교 (응답 시간, 관련성 점수)
- 검색된 문서 분석
- 상세 결과 리포트 생성
- CSV/TXT 형태로 결과 내보내기

### 🔧 사용자 친화적 인터페이스
- 직관적인 React 웹 인터페이스
- 문서 업로드 (PDF, TXT, DOCX 지원)
- 실시간 시스템 상태 모니터링
- 설정값 조정 (청크 크기, 오버랩, Top-K)

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │   FastAPI API   │    │  Google AI API  │
│   (사용자 인터페이스) │◄──►│  (비즈니스 로직)  │◄──►│   (LLM 생성)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   ChromaDB      │
                       │  (벡터 저장소)   │
                       └─────────────────┘
```

## 🚀 빠른 시작

### 1️⃣ 환경 설정

```bash
# 저장소 클론
git clone https://github.com/your-username/rag-optimization-comparison.git
cd rag-optimization-comparison

# Python 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2️⃣ Google AI Studio API 키 설정

```bash
# .env 파일 생성
echo "GOOGLE_API_KEY=your_google_ai_studio_api_key" > .env
```

> 🔑 [Google AI Studio](https://makersuite.google.com/app/apikey)에서 무료 API 키를 발급받으세요.

### 3️⃣ 백엔드 서버 실행

```bash
# FastAPI 서버 시작
python app.py

# 또는 uvicorn 사용
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

서버가 시작되면 `http://localhost:8000`에서 API 문서를 확인할 수 있습니다.

### 4️⃣ 프론트엔드 실행

프론트엔드는 `react.js` 파일의 React 컴포넌트를 사용합니다. 별도의 React 앱을 생성하거나 제공된 HTML 파일을 통해 실행할 수 있습니다.

## 📋 사용 방법

### 1. 문서 업로드
- 웹 인터페이스에서 PDF, TXT, 또는 DOCX 파일을 업로드
- 시스템이 자동으로 문서를 청킹하고 벡터 데이터베이스에 저장

### 2. RAG 방법 선택
- 비교하고 싶은 RAG 최적화 방법들을 선택
- 청크 크기, 오버랩, Top-K 값 조정

### 3. 질문 입력 및 비교 실행
- 문서와 관련된 질문을 입력
- "비교 실행" 버튼을 클릭하여 선택한 방법들의 성능 비교

### 4. 결과 분석
- 각 방법의 응답 시간, 관련성 점수, 검색된 문서 확인
- 성능 요약 통계 및 최적 방법 추천 확인

## 🧪 자동 테스트

시스템의 성능을 자동으로 테스트하고 결과를 파일로 저장할 수 있습니다:

```bash
# 성능 테스트 실행
python rag-test.py
```

테스트 완료 후 다음 파일들이 생성됩니다:
- `rag_comparison_detailed_YYYYMMDD_HHMMSS.txt`: 상세 결과
- `rag_comparison_data_YYYYMMDD_HHMMSS.csv`: 데이터 분석용 CSV
- `rag_summary_YYYYMMDD_HHMMSS.txt`: 요약 보고서

## 🛠️ 기술 스택

### Backend
- **FastAPI**: 고성능 웹 프레임워크
- **LangChain**: LLM 애플리케이션 개발 프레임워크
- **ChromaDB**: 벡터 데이터베이스
- **HuggingFace Transformers**: 임베딩 모델
- **Google AI Studio**: LLM API

### Frontend
- **React**: 사용자 인터페이스
- **Tailwind CSS**: 스타일링
- **Lucide Icons**: 아이콘

### AI/ML
- **Sentence Transformers**: 텍스트 임베딩
- **BM25**: 키워드 기반 검색
- **Gemini Pro**: 텍스트 생성

## 📊 성능 메트릭

시스템은 다음 메트릭으로 RAG 방법들을 평가합니다:

- **응답 시간** (Response Time): 질문부터 답변까지 소요 시간
- **관련성 점수** (Relevance Score): 검색된 문서와 질문의 관련성
- **검색 정확도**: 상위 K개 문서의 품질
- **안정성**: 여러 질문에 대한 성능 일관성

## 🔧 설정 옵션

### 문서 처리 설정
- **Chunk Size**: 문서 분할 크기 (100-4000자)
- **Chunk Overlap**: 청크 간 중복 구간 (0-1000자)
- **Top-K**: 검색할 상위 문서 수 (1-20개)

### 모델 설정
- **임베딩 모델**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM 모델**: `gemini-1.5-flash`
- **온도**: 0.3 (창의성 vs 일관성 조절)

## 🤝 기여하기

1. 이 저장소를 포크합니다
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성합니다

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🔗 관련 링크

- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [LangChain 문서](https://docs.langchain.com/)
- [Google AI Studio](https://makersuite.google.com/)
- [ChromaDB 문서](https://docs.trychroma.com/)

## 📧 문의

질문이나 제안사항이 있으시면 이슈를 생성하거나 연락해 주세요.

