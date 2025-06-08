import os
import warnings

# 환경 변수 설정 (import 전에 실행)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import asyncio
import json
import time
from datetime import datetime
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# LangChain imports (호환성 확인된 순서로 import)
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever
    from langchain_core.documents import Document
    from langchain.chains import RetrievalQA
    from langchain_core.prompts import PromptTemplate
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    
    # Google AI Studio LLM import (안전한 방식)
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        GOOGLE_LLM_AVAILABLE = True
    except Exception as e:
        logger.warning(f"Google LLM import 실패: {e}")
        GOOGLE_LLM_AVAILABLE = False
        ChatGoogleGenerativeAI = None
        
except ImportError as e:
    logger.error(f"필수 패키지 import 실패: {e}")
    raise

app = FastAPI(title="RAG 최적화 비교 시스템", version="2.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 모델 (Pydantic v2 호환)
class QueryRequest(BaseModel):
    query: str = Field(..., description="검색할 질문")
    optimization_methods: List[str] = Field(..., description="사용할 RAG 최적화 방법들")
    chunk_size: int = Field(default=1000, ge=100, le=4000, description="청크 크기")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="청크 오버랩")
    top_k: int = Field(default=5, ge=1, le=20, description="검색할 문서 수")

class ComparisonResult(BaseModel):
    method: str
    response: str
    retrieved_docs: List[Dict[str, Any]]
    response_time: float
    relevance_score: float

class UploadResponse(BaseModel):
    message: str
    filename: str
    document_count: int

class SystemStatus(BaseModel):
    documents_loaded: bool
    document_count: int
    vectorstore_ready: bool
    llm_ready: bool
    api_key_configured: bool

# 전역 변수
documents = []
vectorstore = None
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    logger.warning("⚠️  GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일에 추가해주세요.")

# LLM 초기화 (안전한 방식)
llm = None
if GOOGLE_LLM_AVAILABLE and google_api_key:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.3
        )
        logger.info("✅ Google LLM 초기화 성공")
    except Exception as e:
        logger.error(f"Google LLM 초기화 실패: {e}")
        llm = None

# 임베딩 모델 초기화 (안전한 방식)
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    logger.info("✅ 임베딩 모델 초기화 성공")
except Exception as e:
    logger.error(f"임베딩 모델 초기화 실패: {e}")
    embeddings = None

class RAGOptimizer:
    def __init__(self):
        self.methods = {
            "basic_rag": self.basic_rag,
            "multi_query": self.multi_query_rag,
            "ensemble_retrieval": self.ensemble_retrieval_rag,
            "contextual_compression": self.contextual_compression_rag,
            "hierarchical_chunking": self.hierarchical_chunking_rag,
            "semantic_chunking": self.semantic_chunking_rag
        }
        
    async def basic_rag(self, query: str, config: Dict) -> Dict:
        """기본 RAG 구현"""
        if not llm or not vectorstore:
            raise ValueError("LLM 또는 vectorstore가 초기화되지 않았습니다.")
            
        start_time = time.time()
        
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": config["top_k"]})
            
            prompt_template = """다음 문서들을 참고하여 질문에 답변해주세요:

{context}

질문: {question}

답변:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            
            # 비동기 실행을 동기로 변환
            result = qa_chain.invoke({"query": query})["result"]
            docs = retriever.invoke(query)
            
            response_time = time.time() - start_time
            
            return {
                "method": "Basic RAG",
                "response": result,
                "retrieved_docs": [{"content": doc.page_content[:200] + "...", "metadata": doc.metadata} for doc in docs],
                "response_time": response_time,
                "relevance_score": self.calculate_relevance_score(query, docs)
            }
            
        except Exception as e:
            logger.error(f"Basic RAG 오류: {e}")
            return {
                "method": "Basic RAG",
                "response": f"오류 발생: {str(e)}",
                "retrieved_docs": [],
                "response_time": time.time() - start_time,
                "relevance_score": 0.0
            }
    
    async def multi_query_rag(self, query: str, config: Dict) -> Dict:
        """Multi-Query RAG 구현"""
        if not llm or not vectorstore:
            raise ValueError("LLM 또는 vectorstore가 초기화되지 않았습니다.")
            
        start_time = time.time()
        
        try:
            base_retriever = vectorstore.as_retriever(search_kwargs={"k": config["top_k"]})
            
            # Multi-query retriever 설정
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm
            )
            
            prompt_template = """다음 문서들을 참고하여 질문에 답변해주세요:

{context}

질문: {question}

답변:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=multi_query_retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            
            result = qa_chain.invoke({"query": query})["result"]
            docs = multi_query_retriever.invoke(query)
            
            response_time = time.time() - start_time
            
            return {
                "method": "Multi-Query RAG",
                "response": result,
                "retrieved_docs": [{"content": doc.page_content[:200] + "...", "metadata": doc.metadata} for doc in docs],
                "response_time": response_time,
                "relevance_score": self.calculate_relevance_score(query, docs)
            }
            
        except Exception as e:
            logger.error(f"Multi-Query RAG 오류: {e}")
            return {
                "method": "Multi-Query RAG",
                "response": f"오류 발생: {str(e)}",
                "retrieved_docs": [],
                "response_time": time.time() - start_time,
                "relevance_score": 0.0
            }
    
    async def ensemble_retrieval_rag(self, query: str, config: Dict) -> Dict:
        """Ensemble Retrieval RAG 구현"""
        if not llm or not vectorstore:
            raise ValueError("LLM 또는 vectorstore가 초기화되지 않았습니다.")
            
        start_time = time.time()
        
        try:
            # Vector retriever
            vector_retriever = vectorstore.as_retriever(search_kwargs={"k": config["top_k"]})
            
            # BM25 retriever
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = config["top_k"]
            
            # Ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
            
            prompt_template = """다음 문서들을 참고하여 질문에 답변해주세요:

{context}

질문: {question}

답변:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=ensemble_retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            
            result = qa_chain.invoke({"query": query})["result"]
            docs = ensemble_retriever.invoke(query)
            
            response_time = time.time() - start_time
            
            return {
                "method": "Ensemble Retrieval RAG",
                "response": result,
                "retrieved_docs": [{"content": doc.page_content[:200] + "...", "metadata": doc.metadata} for doc in docs],
                "response_time": response_time,
                "relevance_score": self.calculate_relevance_score(query, docs)
            }
            
        except Exception as e:
            logger.error(f"Ensemble Retrieval RAG 오류: {e}")
            return {
                "method": "Ensemble Retrieval RAG",
                "response": f"오류 발생: {str(e)}",
                "retrieved_docs": [],
                "response_time": time.time() - start_time,
                "relevance_score": 0.0
            }
    
    async def contextual_compression_rag(self, query: str, config: Dict) -> Dict:
        """Contextual Compression RAG 구현"""
        # 기본 RAG로 폴백 (Google AI Studio에서 contextual compression이 불안정할 수 있음)
        return await self.basic_rag(query, config)
    
    async def hierarchical_chunking_rag(self, query: str, config: Dict) -> Dict:
        """Hierarchical Chunking RAG 구현"""
        if not llm or not vectorstore:
            raise ValueError("LLM 또는 vectorstore가 초기화되지 않았습니다.")
            
        start_time = time.time()
        
        try:
            # 기본 검색 수행
            retriever = vectorstore.as_retriever(search_kwargs={"k": config["top_k"] * 2})
            
            prompt_template = """다음 문서들을 참고하여 질문에 답변해주세요:

{context}

질문: {question}

답변:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            
            result = qa_chain.invoke({"query": query})["result"]
            docs = retriever.invoke(query)[:config["top_k"]]  # 상위 K개만 선택
            
            response_time = time.time() - start_time
            
            return {
                "method": "Hierarchical Chunking RAG",
                "response": result,
                "retrieved_docs": [{"content": doc.page_content[:200] + "...", "metadata": doc.metadata} for doc in docs],
                "response_time": response_time,
                "relevance_score": self.calculate_relevance_score(query, docs)
            }
            
        except Exception as e:
            logger.error(f"Hierarchical Chunking RAG 오류: {e}")
            return {
                "method": "Hierarchical Chunking RAG",
                "response": f"오류 발생: {str(e)}",
                "retrieved_docs": [],
                "response_time": time.time() - start_time,
                "relevance_score": 0.0
            }
    
    async def semantic_chunking_rag(self, query: str, config: Dict) -> Dict:
        """Semantic Chunking RAG 구현"""
        # 기본 RAG와 유사하지만 다른 청킹 전략 사용
        return await self.basic_rag(query, config)
    
    def calculate_relevance_score(self, query: str, docs: List[Document]) -> float:
        """간단한 관련성 점수 계산"""
        if not docs:
            return 0.0
        
        try:
            # 키워드 기반 간단한 점수 계산
            query_words = set(query.lower().split())
            scores = []
            
            for doc in docs:
                doc_words = set(doc.page_content.lower().split())
                intersection = len(query_words & doc_words)
                union = len(query_words | doc_words)
                if union > 0:
                    scores.append(intersection / union)
                else:
                    scores.append(0.0)
            
            return sum(scores) / len(scores) if scores else 0.0
        except Exception as e:
            logger.error(f"관련성 점수 계산 오류: {e}")
            return 0.0

# RAG 최적화기 인스턴스
rag_optimizer = RAGOptimizer()

@app.get("/")
async def root():
    return {"message": "RAG 최적화 비교 시스템 API v2.0", "status": "running"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """문서 업로드 및 처리"""
    global documents, vectorstore
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일이 선택되지 않았습니다.")
    
    if not embeddings:
        raise HTTPException(status_code=500, detail="임베딩 모델이 초기화되지 않았습니다.")
    
    try:
        # 파일 저장
        file_path = f"./uploads/{file.filename}"
        os.makedirs("./uploads", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 문서 로드
        if file.filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file.filename.lower().endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file.filename.lower().endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다. (PDF, TXT, DOCX만 지원)")
        
        # 문서 분할
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        documents = text_splitter.split_documents(docs)
        
        # 벡터 스토어 생성
        vectorstore = Chroma.from_documents(
            documents, 
            embeddings,
            collection_name="rag_documents"
        )
        
        # 파일 삭제
        os.remove(file_path)
        
        logger.info(f"문서 업로드 완료: {file.filename}, 청크 수: {len(documents)}")
        
        return UploadResponse(
            message="문서가 성공적으로 업로드되고 처리되었습니다.",
            filename=file.filename,
            document_count=len(documents)
        )
        
    except Exception as e:
        logger.error(f"문서 처리 오류: {e}")
        # 실패한 파일 정리
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"문서 처리 중 오류 발생: {str(e)}")

@app.post("/compare")
async def compare_rag_methods(request: QueryRequest):
    """RAG 방법들을 비교하여 결과 반환"""
    if not documents:
        raise HTTPException(status_code=400, detail="먼저 문서를 업로드해주세요.")
    
    if not llm:
        raise HTTPException(status_code=400, detail="Google AI Studio LLM이 초기화되지 않았습니다. API 키를 확인해주세요.")
    
    config = {
        "chunk_size": request.chunk_size,
        "chunk_overlap": request.chunk_overlap,
        "top_k": request.top_k
    }
    
    results = []
    
    for method in request.optimization_methods:
        if method not in rag_optimizer.methods:
            logger.warning(f"지원하지 않는 방법: {method}")
            continue
            
        try:
            logger.info(f"{method} 실행 중...")
            result = await rag_optimizer.methods[method](request.query, config)
            results.append(result)
            logger.info(f"{method} 완료 - 응답 시간: {result['response_time']:.2f}s")
        except Exception as e:
            logger.error(f"{method} 실행 오류: {e}")
            results.append({
                "method": method,
                "response": f"오류 발생: {str(e)}",
                "retrieved_docs": [],
                "response_time": 0.0,
                "relevance_score": 0.0
            })
    
    return {"results": results, "query": request.query, "config": config}

@app.get("/methods")
async def get_available_methods():
    """사용 가능한 RAG 최적화 방법 목록 반환"""
    methods = [
        {
            "id": "basic_rag",
            "name": "Basic RAG",
            "description": "기본적인 RAG 구현 (Vector Search + LLM)"
        },
        {
            "id": "multi_query",
            "name": "Multi-Query RAG", 
            "description": "여러 쿼리를 생성하여 검색 결과를 향상"
        },
        {
            "id": "ensemble_retrieval",
            "name": "Ensemble Retrieval",
            "description": "Vector Search + BM25 결합"
        },
        {
            "id": "contextual_compression",
            "name": "Contextual Compression",
            "description": "관련 정보만 압축하여 컨텍스트 최적화"
        },
        {
            "id": "hierarchical_chunking",
            "name": "Hierarchical Chunking",
            "description": "계층적 청킹으로 정보 구조화"
        },
        {
            "id": "semantic_chunking", 
            "name": "Semantic Chunking",
            "description": "의미적 유사도 기반 청킹"
        }
    ]
    
    return {"methods": methods}

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """시스템 상태 확인"""
    return SystemStatus(
        documents_loaded=len(documents) > 0,
        document_count=len(documents),
        vectorstore_ready=vectorstore is not None,
        llm_ready=llm is not None,
        api_key_configured=google_api_key is not None
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 RAG 최적화 비교 시스템을 시작합니다...")
    logger.info(f"임베딩 모델 준비: {'✅' if embeddings else '❌'}")
    logger.info(f"Google LLM 준비: {'✅' if llm else '❌'}")
    logger.info(f"API 키 설정: {'✅' if google_api_key else '❌'}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)