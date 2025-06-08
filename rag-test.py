import requests
import json
import time
import os
from datetime import datetime
import csv

# 서버 URL
BASE_URL = "http://localhost:8000"

def save_results_to_files(all_results, summary_stats):
    """결과를 여러 형태의 파일로 저장"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 상세 결과를 텍스트 파일로 저장
    txt_filename = f"rag_comparison_detailed_{timestamp}.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RAG 성능 비교 상세 결과\n")
        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, query_result in enumerate(all_results, 1):
            f.write(f"질문 {i}: {query_result['query']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"전체 처리 시간: {query_result['total_time']:.2f}초\n\n")
            
            # 결과를 관련성 점수 기준으로 정렬
            sorted_results = sorted(query_result['results'], 
                                  key=lambda x: x['relevance_score'], reverse=True)
            
            for j, result in enumerate(sorted_results, 1):
                f.write(f"{j}위. {result['method']}\n")
                f.write(f"   응답 시간: {result['response_time']:.2f}초\n")
                f.write(f"   관련성 점수: {result['relevance_score']:.3f}\n")
                f.write(f"   검색 문서 수: {len(result['retrieved_docs'])}\n")
                f.write(f"   답변: {result['response']}\n")
                
                # 검색된 문서들
                f.write(f"   검색된 문서들:\n")
                for k, doc in enumerate(result['retrieved_docs'], 1):
                    f.write(f"      문서 {k}: {doc['content']}\n")
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
        
        # 요약 통계
        f.write("📊 전체 요약 통계\n")
        f.write("=" * 80 + "\n")
        for method, stats in summary_stats.items():
            f.write(f"\n{method}:\n")
            f.write(f"   평균 응답 시간: {stats['avg_response_time']:.2f}초\n")
            f.write(f"   평균 관련성 점수: {stats['avg_relevance_score']:.3f}\n")
            f.write(f"   최고 관련성 점수: {stats['max_relevance_score']:.3f}\n")
            f.write(f"   최저 관련성 점수: {stats['min_relevance_score']:.3f}\n")
    
    # 2. CSV 파일로 저장 (데이터 분석용)
    csv_filename = f"rag_comparison_data_{timestamp}.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 헤더
        writer.writerow([
            "질문_번호", "질문", "방법", "응답_시간(초)", "관련성_점수", 
            "검색_문서_수", "답변_길이", "순위"
        ])
        
        # 데이터
        for i, query_result in enumerate(all_results, 1):
            sorted_results = sorted(query_result['results'], 
                                  key=lambda x: x['relevance_score'], reverse=True)
            
            for j, result in enumerate(sorted_results, 1):
                writer.writerow([
                    i,
                    query_result['query'],
                    result['method'],
                    result['response_time'],
                    result['relevance_score'],
                    len(result['retrieved_docs']),
                    len(result['response']),
                    j
                ])
    
    # 3. 요약 통계를 별도 텍스트 파일로 저장
    summary_filename = f"rag_summary_{timestamp}.txt"
    with open(summary_filename, "w", encoding="utf-8") as f:
        f.write("RAG 성능 비교 요약 보고서\n")
        f.write("=" * 50 + "\n")
        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"총 질문 수: {len(all_results)}\n")
        f.write(f"비교한 방법 수: {len(summary_stats)}\n\n")
        
        # 방법별 순위
        method_rankings = {}
        for query_result in all_results:
            sorted_results = sorted(query_result['results'], 
                                  key=lambda x: x['relevance_score'], reverse=True)
            for rank, result in enumerate(sorted_results, 1):
                method = result['method']
                if method not in method_rankings:
                    method_rankings[method] = []
                method_rankings[method].append(rank)
        
        # 평균 순위 계산
        f.write("📈 방법별 평균 순위 (낮을수록 좋음):\n")
        f.write("-" * 30 + "\n")
        avg_rankings = {}
        for method, ranks in method_rankings.items():
            avg_rank = sum(ranks) / len(ranks)
            avg_rankings[method] = avg_rank
            f.write(f"{method}: {avg_rank:.2f}위\n")
        
        # 최고 성능 방법
        best_method = min(avg_rankings.items(), key=lambda x: x[1])
        f.write(f"\n🏆 최고 성능 방법: {best_method[0]} (평균 {best_method[1]:.2f}위)\n")
        
        # 상세 통계
        f.write("\n📊 상세 통계:\n")
        f.write("-" * 30 + "\n")
        for method, stats in summary_stats.items():
            f.write(f"\n{method}:\n")
            f.write(f"   평균 응답 시간: {stats['avg_response_time']:.2f}초\n")
            f.write(f"   평균 관련성 점수: {stats['avg_relevance_score']:.3f}\n")
            f.write(f"   성능 안정성: {stats['relevance_std']:.3f} (낮을수록 안정)\n")
        
        # 추천사항
        f.write("\n💡 추천사항:\n")
        f.write("-" * 30 + "\n")
        
        fastest_method = min(summary_stats.items(), 
                           key=lambda x: x[1]['avg_response_time'])
        most_relevant = max(summary_stats.items(), 
                          key=lambda x: x[1]['avg_relevance_score'])
        most_stable = min(summary_stats.items(), 
                        key=lambda x: x[1]['relevance_std'])
        
        f.write(f"가장 빠른 방법: {fastest_method[0]} ({fastest_method[1]['avg_response_time']:.2f}초)\n")
        f.write(f"가장 관련성 높은 방법: {most_relevant[0]} ({most_relevant[1]['avg_relevance_score']:.3f})\n")
        f.write(f"가장 안정적인 방법: {most_stable[0]} (표준편차: {most_stable[1]['relevance_std']:.3f})\n")
    
    return txt_filename, csv_filename, summary_filename

def calculate_summary_statistics(all_results):
    """전체 결과에서 요약 통계 계산"""
    
    method_stats = {}
    
    for query_result in all_results:
        for result in query_result['results']:
            method = result['method']
            
            if method not in method_stats:
                method_stats[method] = {
                    'response_times': [],
                    'relevance_scores': []
                }
            
            method_stats[method]['response_times'].append(result['response_time'])
            method_stats[method]['relevance_scores'].append(result['relevance_score'])
    
    # 통계 계산
    summary_stats = {}
    for method, data in method_stats.items():
        response_times = data['response_times']
        relevance_scores = data['relevance_scores']
        
        summary_stats[method] = {
            'avg_response_time': sum(response_times) / len(response_times),
            'avg_relevance_score': sum(relevance_scores) / len(relevance_scores),
            'max_relevance_score': max(relevance_scores),
            'min_relevance_score': min(relevance_scores),
            'relevance_std': (sum((x - sum(relevance_scores)/len(relevance_scores))**2 
                                for x in relevance_scores) / len(relevance_scores))**0.5
        }
    
    return summary_stats

def test_rag_comparison():
    print("🧪 RAG 성능 비교 테스트 시작")
    
    all_results = []  # 모든 결과를 저장할 리스트
    
    # 1. 시스템 상태 확인
    print("\n1️⃣ 시스템 상태 확인...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        status = response.json()
        print(f"문서 로드됨: {status['documents_loaded']}")
        print(f"문서 수: {status['document_count']}")
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        return
    
    # 2. 문서가 없다면 테스트 문서 업로드
    if not status['documents_loaded']:
        print("\n2️⃣ 테스트 문서 업로드...")
        
        # 더 풍부한 테스트 문서 생성
        test_content = """
        인공지능과 머신러닝 완전 가이드
        
        1. 인공지능(AI) 개요
        인공지능(AI)은 인간의 지능을 모방하는 컴퓨터 시스템입니다.
        1950년 앨런 튜링이 제안한 튜링 테스트부터 시작되어 현재까지 발전해왔습니다.
        
        2. 주요 AI 기술들
        
        2.1 머신러닝 (Machine Learning)
        - 정의: 데이터로부터 패턴을 학습하여 예측하는 기술
        - 종류: 지도학습, 비지도학습, 강화학습
        - 활용: 추천 시스템, 이미지 분류, 음성 인식
        
        2.2 딥러닝 (Deep Learning)
        - 정의: 인공신경망을 사용한 머신러닝의 한 분야
        - 특징: 다층 신경망, 자동 특성 추출
        - 활용: 자연어처리, 컴퓨터 비전, 자율주행
        
        2.3 자연어처리 (NLP)
        - 정의: 인간의 언어를 컴퓨터가 이해하고 처리하는 기술
        - 기술: 토큰화, 품사 태깅, 개체명 인식, 감정 분석
        - 활용: 번역, 챗봇, 문서 요약, 질의응답
        
        2.4 컴퓨터 비전
        - 정의: 이미지와 비디오를 분석하고 이해하는 기술
        - 기술: 객체 검출, 이미지 분할, 얼굴 인식
        - 활용: 의료 진단, 자율주행, 보안 시스템
        
        3. RAG (Retrieval-Augmented Generation)
        
        3.1 RAG란?
        RAG는 검색 증강 생성 기술로, 외부 지식베이스에서 관련 정보를 검색한 후 
        그 정보를 바탕으로 답변을 생성하는 AI 기술입니다.
        
        3.2 RAG의 구성요소
        - 문서 임베딩: 텍스트를 벡터로 변환
        - 벡터 데이터베이스: 임베딩된 문서 저장
        - 검색기: 질문과 관련된 문서 검색
        - 생성기: 검색된 문서를 바탕으로 답변 생성
        
        3.3 RAG의 장점
        - 최신 정보 반영 가능
        - 환각(Hallucination) 현상 감소
        - 투명한 정보 출처 제공
        - 도메인 특화 지식 활용
        
        3.4 RAG 최적화 방법
        - Basic RAG: 기본적인 벡터 검색
        - Multi-Query RAG: 다중 쿼리로 검색 다양성 확보
        - Ensemble Retrieval: 벡터 검색과 키워드 검색 결합
        - Contextual Compression: 관련 정보만 선별
        - Hierarchical Chunking: 계층적 문서 분할
        - Semantic Chunking: 의미 기반 문서 분할
        
        4. 머신러닝 vs 딥러닝 비교
        
        4.1 공통점
        - 둘 다 데이터로부터 학습
        - 예측과 분류 작업 수행
        - 통계적 모델링 기반
        
        4.2 차이점
        머신러닝:
        - 상대적으로 단순한 알고리즘
        - 특성 공학이 중요
        - 소량 데이터로도 학습 가능
        - 해석이 용이
        
        딥러닝:
        - 복잡한 신경망 구조
        - 자동 특성 추출
        - 대용량 데이터 필요
        - 높은 성능, 낮은 해석성
        
        5. AI의 미래 전망
        - 자율주행차의 상용화
        - 의료 AI의 확산
        - 창작 AI의 발전
        - 범용 인공지능(AGI) 연구
        """
        
        with open("comprehensive_ai_guide.txt", "w", encoding="utf-8") as f:
            f.write(test_content)
        
        # 파일 업로드
        with open("comprehensive_ai_guide.txt", "rb") as f:
            files = {"file": ("comprehensive_ai_guide.txt", f, "text/plain")}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 업로드 성공: {result['filename']}, 청크 수: {result['document_count']}")
            else:
                print(f"❌ 업로드 실패: {response.text}")
                return
    
    # 3. RAG 방법들 비교
    print("\n3️⃣ RAG 방법들 성능 비교...")
    
    test_queries = [
        "인공지능의 주요 기술은 무엇인가요?",
        "RAG의 장점을 설명해주세요.",
        "머신러닝과 딥러닝의 차이점은 무엇인가요?",
        "자연어처리의 주요 기술들을 나열해주세요.",
        "RAG 최적화 방법들에는 어떤 것들이 있나요?"
    ]
    
    methods = ["basic_rag", "multi_query", "ensemble_retrieval", "hierarchical_chunking"]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 질문 {i}: {query}")
        
        # 비교 요청
        comparison_data = {
            "query": query,
            "optimization_methods": methods,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k": 3
        }
        
        start_time = time.time()
        try:
            response = requests.post(f"{BASE_URL}/compare", json=comparison_data)
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                results = response.json()["results"]
                
                # 결과 저장
                query_result = {
                    "query": query,
                    "results": results,
                    "total_time": total_time
                }
                all_results.append(query_result)
                
                print(f"⏱️ 전체 처리 시간: {total_time:.2f}초")
                print("\n📊 결과 비교:")
                
                # 결과 정렬 (관련성 점수 기준)
                sorted_results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
                
                for j, result in enumerate(sorted_results, 1):
                    print(f"\n{j}위. {result['method']}")
                    print(f"   ⏱️ 응답시간: {result['response_time']:.2f}초")
                    print(f"   📈 관련성 점수: {result['relevance_score']:.3f}")
                    print(f"   📄 검색 문서 수: {len(result['retrieved_docs'])}")
                    print(f"   💬 답변: {result['response'][:150]}...")
                    
            else:
                print(f"❌ 비교 실패: {response.text}")
                
        except Exception as e:
            print(f"❌ 요청 오류: {e}")
    
    # 4. 결과 분석 및 저장
    if all_results:
        print("\n4️⃣ 결과 분석 및 파일 저장...")
        
        # 요약 통계 계산
        summary_stats = calculate_summary_statistics(all_results)
        
        # 파일 저장
        txt_file, csv_file, summary_file = save_results_to_files(all_results, summary_stats)
        
        print("\n📁 저장된 파일들:")
        print(f"   📄 상세 결과: {txt_file}")
        print(f"   📊 데이터 CSV: {csv_file}")
        print(f"   📋 요약 보고서: {summary_file}")
        
        # 간단한 최종 요약 출력
        print("\n🏆 최종 요약:")
        print("-" * 40)
        
        # 평균 순위 계산
        method_rankings = {}
        for query_result in all_results:
            sorted_results = sorted(query_result['results'], 
                                  key=lambda x: x['relevance_score'], reverse=True)
            for rank, result in enumerate(sorted_results, 1):
                method = result['method']
                if method not in method_rankings:
                    method_rankings[method] = []
                method_rankings[method].append(rank)
        
        for method, ranks in method_rankings.items():
            avg_rank = sum(ranks) / len(ranks)
            avg_time = summary_stats[method]['avg_response_time']
            avg_relevance = summary_stats[method]['avg_relevance_score']
            print(f"{method}:")
            print(f"   평균 순위: {avg_rank:.1f}위")
            print(f"   평균 응답시간: {avg_time:.2f}초")
            print(f"   평균 관련성: {avg_relevance:.3f}")
            print()
    
    print("\n🎉 RAG 성능 비교 테스트 완료!")
    print("생성된 파일들을 확인하여 상세한 분석 결과를 확인하세요.")

if __name__ == "__main__":
    test_rag_comparison()