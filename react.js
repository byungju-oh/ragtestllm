import React, { useState, useEffect } from 'react';
import { Upload, Play, FileText, Clock, TrendingUp, CheckCircle, AlertCircle, Loader } from 'lucide-react';

const RAGComparisonApp = () => {
  const [file, setFile] = useState(null);
  const [query, setQuery] = useState('');
  const [selectedMethods, setSelectedMethods] = useState([]);
  const [availableMethods, setAvailableMethods] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState({});
  const [config, setConfig] = useState({
    chunk_size: 1000,
    chunk_overlap: 200,
    top_k: 5
  });

  const API_BASE = 'http://localhost:8000';

  useEffect(() => {
    fetchAvailableMethods();
    fetchSystemStatus();
  }, []);

  const fetchAvailableMethods = async () => {
    try {
      const response = await fetch(`${API_BASE}/methods`);
      const data = await response.json();
      setAvailableMethods(data.methods);
    } catch (error) {
      console.error('메서드 로딩 실패:', error);
    }
  };

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/status`);
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('상태 확인 실패:', error);
    }
  };

  const handleFileUpload = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setUploadLoading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        alert(`${result.message} (${result.document_count}개 청크 생성)`);
        await fetchSystemStatus();
      } else {
        alert('파일 업로드 실패');
      }
    } catch (error) {
      console.error('업로드 오류:', error);
      alert('업로드 중 오류가 발생했습니다.');
    } finally {
      setUploadLoading(false);
    }
  };

  const handleMethodToggle = (methodId) => {
    setSelectedMethods(prev => 
      prev.includes(methodId) 
        ? prev.filter(id => id !== methodId)
        : [...prev, methodId]
    );
  };

  const handleCompare = async () => {
    if (!query.trim() || selectedMethods.length === 0) {
      alert('질문과 최소 하나의 방법을 선택해주세요.');
      return;
    }

    setLoading(true);
    setResults([]);

    try {
      const response = await fetch(`${API_BASE}/compare`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          optimization_methods: selectedMethods,
          chunk_size: config.chunk_size,
          chunk_overlap: config.chunk_overlap,
          top_k: config.top_k
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setResults(data.results);
      } else {
        alert('비교 요청 실패');
      }
    } catch (error) {
      console.error('비교 오류:', error);
      alert('비교 중 오류가 발생했습니다.');
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (seconds) => {
    return `${seconds.toFixed(2)}초`;
  };

  const getScoreColor = (score) => {
    if (score >= 0.7) return 'text-green-600';
    if (score >= 0.4) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getBestMethod = () => {
    if (results.length === 0) return null;
    return results.reduce((best, current) => 
      current.relevance_score > best.relevance_score ? current : best
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* 헤더 */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            🔍 RAG 최적화 방법 비교 시스템
          </h1>
          <p className="text-lg text-gray-600">
            FastAPI + LangChain + Google AI Studio로 구현된 RAG 성능 비교 도구
          </p>
        </div>

        {/* 시스템 상태 */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <CheckCircle className="mr-2 text-green-500" size={20} />
            시스템 상태
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${systemStatus.documents_loaded ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm">문서 로드됨</span>
            </div>
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${systemStatus.vectorstore_ready ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm">벡터스토어 준비</span>
            </div>
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${systemStatus.llm_ready ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm">LLM 준비</span>
            </div>
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${systemStatus.api_key_configured ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm">API 키 설정</span>
            </div>
          </div>
          {systemStatus.document_count > 0 && (
            <p className="text-sm text-gray-600 mt-2">
              총 {systemStatus.document_count}개의 문서 청크가 로드되었습니다.
            </p>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* 컨트롤 패널 */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md p-6 sticky top-4">
              {/* 파일 업로드 */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold mb-3 flex items-center">
                  <Upload className="mr-2 text-blue-500" size={20} />
                  문서 업로드
                </h3>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-blue-500 transition-colors">
                  <input
                    type="file"
                    accept=".pdf,.txt,.docx"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="file-upload"
                    disabled={uploadLoading}
                  />
                  <label htmlFor="file-upload" className="cursor-pointer">
                    {uploadLoading ? (
                      <div className="flex items-center justify-center">
                        <Loader className="animate-spin mr-2" size={20} />
                        업로드 중...
                      </div>
                    ) : (
                      <>
                        <FileText className="mx-auto mb-2 text-gray-400" size={48} />
                        <p className="text-sm text-gray-600">
                          PDF, TXT, DOCX 파일을 선택하세요
                        </p>
                        {file && (
                          <p className="text-xs text-blue-600 mt-2">
                            선택됨: {file.name}
                          </p>
                        )}
                      </>
                    )}
                  </label>
                </div>
              </div>

              {/* 질문 입력 */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold mb-3">질문</h3>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="문서에 대한 질문을 입력하세요..."
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows={4}
                />
              </div>

              {/* RAG 방법 선택 */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold mb-3">RAG 최적화 방법</h3>
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {availableMethods.map((method) => (
                    <div
                      key={method.id}
                      className={`p-3 border rounded-lg cursor-pointer transition-all ${
                        selectedMethods.includes(method.id)
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-300 hover:border-blue-300'
                      }`}
                      onClick={() => handleMethodToggle(method.id)}
                    >
                      <div className="flex items-center">
                        <input
                          type="checkbox"
                          checked={selectedMethods.includes(method.id)}
                          onChange={() => handleMethodToggle(method.id)}
                          className="mr-3"
                        />
                        <div>
                          <p className="font-medium text-sm">{method.name}</p>
                          <p className="text-xs text-gray-600">{method.description}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* 설정 */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold mb-3">설정</h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      청크 크기: {config.chunk_size}
                    </label>
                    <input
                      type="range"
                      min="500"
                      max="2000"
                      step="100"
                      value={config.chunk_size}
                      onChange={(e) => setConfig({...config, chunk_size: parseInt(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      청크 오버랩: {config.chunk_overlap}
                    </label>
                    <input
                      type="range"
                      min="50"
                      max="500"
                      step="50"
                      value={config.chunk_overlap}
                      onChange={(e) => setConfig({...config, chunk_overlap: parseInt(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      검색 결과 수 (Top-K): {config.top_k}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      step="1"
                      value={config.top_k}
                      onChange={(e) => setConfig({...config, top_k: parseInt(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                </div>
              </div>

              {/* 비교 실행 버튼 */}
              <button
                onClick={handleCompare}
                disabled={loading || !systemStatus.documents_loaded || selectedMethods.length === 0}
                className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
              >
                {loading ? (
                  <>
                    <Loader className="animate-spin mr-2" size={20} />
                    비교 중...
                  </>
                ) : (
                  <>
                    <Play className="mr-2" size={20} />
                    비교 실행
                  </>
                )}
              </button>
            </div>
          </div>

          {/* 결과 패널 */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-2xl font-semibold mb-6 flex items-center">
                <TrendingUp className="mr-2 text-green-500" size={24} />
                비교 결과
              </h2>

              {loading && (
                <div className="text-center py-12">
                  <Loader className="animate-spin mx-auto mb-4 text-blue-500" size={48} />
                  <p className="text-lg text-gray-600">RAG 방법들을 비교하고 있습니다...</p>
                  <p className="text-sm text-gray-500 mt-2">선택된 방법: {selectedMethods.length}개</p>
                </div>
              )}

              {!loading && results.length === 0 && (
                <div className="text-center py-12">
                  <AlertCircle className="mx-auto mb-4 text-gray-400" size={48} />
                  <p className="text-lg text-gray-600">비교할 결과가 없습니다.</p>
                  <p className="text-sm text-gray-500 mt-2">
                    문서를 업로드하고 질문과 방법을 선택한 후 비교를 실행하세요.
                  </p>
                </div>
              )}

              {results.length > 0 && (
                <>
                  {/* 성능 요약 */}
                  <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                    <h3 className="text-lg font-semibold mb-3">성능 요약</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="text-center">
                        <p className="text-2xl font-bold text-blue-600">
                          {getBestMethod()?.method}
                        </p>
                        <p className="text-sm text-gray-600">최고 관련성 점수</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-green-600">
                          {formatTime(Math.min(...results.map(r => r.response_time)))}
                        </p>
                        <p className="text-sm text-gray-600">최단 응답 시간</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-purple-600">
                          {(results.reduce((sum, r) => sum + r.relevance_score, 0) / results.length).toFixed(3)}
                        </p>
                        <p className="text-sm text-gray-600">평균 관련성 점수</p>
                      </div>
                    </div>
                  </div>

                  {/* 상세 결과 */}
                  <div className="space-y-6">
                    {results.map((result, index) => (
                      <div key={index} className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
                        <div className="flex items-center justify-between mb-4">
                          <h3 className="text-xl font-semibold text-gray-800">
                            {result.method}
                          </h3>
                          <div className="flex items-center space-x-4">
                            <div className="flex items-center">
                              <Clock className="mr-1 text-blue-500" size={16} />
                              <span className="text-sm font-medium">{formatTime(result.response_time)}</span>
                            </div>
                            <div className={`text-sm font-bold ${getScoreColor(result.relevance_score)}`}>
                              관련성: {result.relevance_score.toFixed(3)}
                            </div>
                          </div>
                        </div>

                        <div className="mb-4">
                          <h4 className="font-semibold text-gray-700 mb-2">응답:</h4>
                          <div className="bg-gray-50 p-4 rounded-lg">
                            <p className="text-gray-800 leading-relaxed">{result.response}</p>
                          </div>
                        </div>

                        <div>
                          <h4 className="font-semibold text-gray-700 mb-2">
                            검색된 문서 ({result.retrieved_docs.length}개):
                          </h4>
                          <div className="space-y-2 max-h-40 overflow-y-auto">
                            {result.retrieved_docs.map((doc, docIndex) => (
                              <div key={docIndex} className="bg-blue-50 p-3 rounded text-sm">
                                <p className="text-gray-700">{doc.content}</p>
                                {doc.metadata && Object.keys(doc.metadata).length > 0 && (
                                  <p className="text-xs text-gray-500 mt-1">
                                    메타데이터: {JSON.stringify(doc.metadata)}
                                  </p>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* 도움말 */}
        <div className="mt-8 bg-blue-50 border-l-4 border-blue-500 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-blue-800 mb-3">사용 방법</h3>
          <ol className="list-decimal list-inside space-y-2 text-blue-700">
            <li><strong>Google AI Studio API 키 설정:</strong> .env 파일에 GOOGLE_API_KEY를 추가하세요.</li>
            <li><strong>문서 업로드:</strong> PDF, TXT, DOCX 파일을 업로드하여 RAG 시스템에 지식을 제공하세요.</li>
            <li><strong>질문 입력:</strong> 업로드한 문서에 대한 질문을 입력하세요.</li>
            <li><strong>방법 선택:</strong> 비교하고 싶은 RAG 최적화 방법들을 선택하세요.</li>
            <li><strong>설정 조정:</strong> 청크 크기, 오버랩, Top-K 값을 조정하여 성능을 최적화하세요.</li>
            <li><strong>비교 실행:</strong> 선택한 방법들의 성능을 비교하고 결과를 분석하세요.</li>
          </ol>
        </div>

        {/* 푸터 */}
        <div className="mt-8 text-center text-gray-600">
          <p>FastAPI + LangChain + Google AI Studio로 구현된 RAG 최적화 비교 시스템</p>
          <p className="text-sm mt-2">
            백엔드: <code>python main.py</code> | 
            프론트엔드: <code>npm start</code> | 
            API 문서: <a href="http://localhost:8000/docs" className="text-blue-500 hover:underline">http://localhost:8000/docs</a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default RAGComparisonApp;