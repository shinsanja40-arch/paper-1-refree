# 사용 가이드 (Usage Guide)

## 🎯 목적

이 코드는 논문 "Breaking the Data Wall: High-Fidelity Knowledge Synthesis and Self-Evolving AI via Referee-Mediated Discourse"의 실험을 **완전히 재현 가능**하도록 만든 통합 구현입니다.

## 📦 포함된 파일

```
referee-mediated-discourse/
├── referee_mediated_discourse.py  # 메인 실험 코드
├── requirements.txt               # Python 패키지 의존성
├── .env.example                   # API 키 템플릿
├── README.md                      # 영문 문서
├── USAGE_GUIDE.md                # 이 파일 (한글 가이드)
├── quickstart.sh                  # 빠른 시작 스크립트
├── Dockerfile                     # Docker 컨테이너 설정
├── docker-compose.yml             # Docker Compose 설정
└── .gitignore                     # Git 무시 파일 목록
```

## 🚀 빠른 시작 (3가지 방법)

### 방법 1: 자동 스크립트 사용 (가장 쉬움)

```bash
# 1. 스크립트 실행
./quickstart.sh

# 2. 화면의 지시를 따라 API 키 설정
# 3. 실험 선택 (1 또는 2)
```

### 방법 2: 수동 설치

```bash
# 1. 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. API 키 설정
cp .env.example .env
# .env 파일을 편집하여 실제 API 키 입력

# 4. 실험 실행
python referee_mediated_discourse.py --experiment nuclear_energy --seed 42
```

### 방법 3: Docker 사용 (환경 격리)

```bash
# 1. Docker 이미지 빌드
docker build -t referee-debate:latest .

# 2. 실험 실행
docker run \
  -e ANTHROPIC_API_KEY="your-key" \
  -e OPENAI_API_KEY="your-key" \
  -e GOOGLE_API_KEY="your-key" \
  -v $(pwd)/outputs:/app/outputs \
  referee-debate:latest \
  --experiment nuclear_energy --seed 42
```

## 🔑 API 키 발급 방법

### 1. Anthropic Claude API
- https://console.anthropic.com 접속
- 계정 생성/로그인
- API Keys 메뉴에서 키 생성
- 비용: 종량제 (pay-as-you-go)

### 2. OpenAI GPT API
- https://platform.openai.com 접속
- 계정 생성/로그인
- API keys 메뉴에서 키 생성
- 비용: 종량제

### 3. Google Gemini API
- https://ai.google.dev 접속
- 프로젝트 생성
- API key 생성
- 비용: 무료 티어 있음 (제한적)

## 🧪 실험 실행

### 사용 가능한 실험

1. **nuclear_energy**: 원자력 vs 재생에너지 토론
2. **good_vs_evil**: 인간 본성에 대한 철학적 토론

### 명령어 형식

```bash
python referee_mediated_discourse.py \
  --experiment [nuclear_energy|good_vs_evil] \
  --seed [난수 시드]
```

### 예시

```bash
# 원자력 에너지 토론 (시드 42)
python referee_mediated_discourse.py --experiment nuclear_energy --seed 42

# 철학 토론 (시드 123)
python referee_mediated_discourse.py --experiment good_vs_evil --seed 123
```

## 📊 출력 파일 설명

실험 실행 후 `outputs/` 디렉토리에 다음 파일들이 생성됩니다:

```
outputs/nuclear_energy_2025-01-29T10-30-45/
├── config.json                      # 실험 설정
├── full_transcript.json             # 전체 대화 로그
├── hallucination_annotations.json   # 환각 탐지 결과
├── metrics.json                     # 정량적 지표
└── REPORT.md                        # 요약 보고서
```

### config.json
- 실험의 모든 파라미터
- 사용된 모델, 온도, 프롬프트 등
- 재현을 위해 필요한 모든 정보

### full_transcript.json
```json
[
  {
    "turn_number": 1,
    "agent_role": "debater_a",
    "agent_name": "Nuclear Advocate",
    "model": "claude-3-5-sonnet-20241022",
    "timestamp": "2025-01-29T10:30:45.123456",
    "content": "원자력 에너지는...",
    "tokens_used": 450,
    "latency_ms": 1234.56,
    "metadata": {...}
  }
]
```

### hallucination_annotations.json
```json
[
  {
    "turn_number": 2,
    "sentence_id": "turn_2_ref",
    "claim": "Agent가 주장한 내용...",
    "is_hallucination": false,
    "severity": "correct",
    "evidence": ["출처 URL"],
    "annotator_notes": "Referee의 평가"
  }
]
```

### metrics.json
```json
{
  "total_turns": 20,
  "debater_turns": 10,
  "referee_interventions": 10,
  "hallucination_rate": 0.02,
  "correction_rate": 0.98,
  "factual_errors": 0,
  "unverifiable_claims": 2,
  "misleading_claims": 0
}
```

## 🔬 논문 재현을 위한 체크리스트

실험 결과를 논문에서 제시한 결과와 비교할 때:

- [ ] **동일한 시드 사용**: `--seed 42`
- [ ] **동일한 모델 버전**: config.json에서 확인
- [ ] **동일한 실험 설정**: max_turns, temperature 등
- [ ] **metrics.json 비교**: hallucination_rate, correction_rate
- [ ] **full_transcript.json 검토**: 실제 대화 내용 확인

## 🛠️ 커스터마이징

### 새로운 실험 추가

`referee_mediated_discourse.py` 파일을 수정:

```python
# 1. 시스템 프롬프트 추가 (라인 ~150)
SYSTEM_PROMPTS["debater_climate_pro"] = """
당신은 기후변화 대응을 주장하는 전문가입니다...
"""

# 2. 실험 설정 추가 (라인 ~600)
"climate_debate": ExperimentConfig(
    experiment_id=f"climate_debate_{self.timestamp}",
    topic="Climate Change Action",
    description="기후변화 대응 정책 토론",
    max_turns=10,
    # ... 나머지 설정
)

# 3. 초기 프롬프트 추가 (라인 ~750)
initial_prompts["climate_debate"] = """
기후변화 대응의 시급성에 대해 토론하세요...
"""
```

### 파라미터 조정

```python
# 토론 길이 조정
max_turns=20  # 기본값: 10

# 교착 상태 감지 민감도
deadlock_threshold=5  # 기본값: 3 (높을수록 더 관대)

# AI 모델 창의성
temperature=0.9  # 기본값: 0.7 (높을수록 더 창의적)

# 응답 길이
max_tokens=2000  # 기본값: 1000
```

### 다른 모델 사용

```python
agent_a=AgentConfig(
    # Claude 다른 버전
    model="claude-3-opus-20240229",  # Opus (더 강력)
    # 또는
    model="claude-3-haiku-20240307",  # Haiku (더 빠름)
    
    # GPT 다른 버전
    model="gpt-4-turbo-preview",
    # 또는
    model="gpt-3.5-turbo",  # 더 저렴
)
```

## 📈 결과 분석

### Python에서 결과 로드

```python
import json

# 메트릭 로드
with open('outputs/.../metrics.json') as f:
    metrics = json.load(f)
    
print(f"환각률: {metrics['hallucination_rate']:.2%}")
print(f"교정률: {metrics['correction_rate']:.2%}")

# 전체 대화 로드
with open('outputs/.../full_transcript.json') as f:
    transcript = json.load(f)
    
for turn in transcript[:5]:  # 처음 5턴 출력
    print(f"\n=== {turn['agent_name']} ===")
    print(turn['content'][:200])
```

### 여러 실험 비교

```bash
# 다른 시드로 3번 실행
python referee_mediated_discourse.py --experiment nuclear_energy --seed 42
python referee_mediated_discourse.py --experiment nuclear_energy --seed 123
python referee_mediated_discourse.py --experiment nuclear_energy --seed 999

# 결과 비교 스크립트
python compare_results.py outputs/nuclear_energy_*
```

## ⚠️ 주의사항

### API 비용
- Claude: ~$0.003/1K tokens (입력), ~$0.015/1K tokens (출력)
- GPT-4: ~$0.01/1K tokens (입력), ~$0.03/1K tokens (출력)
- Gemini: 일일 무료 한도 있음

**예상 비용 (10턴 토론)**:
- Agent A (Claude): ~$0.50
- Agent B (GPT-4): ~$1.00
- Referee (Gemini): 무료 (한도 내)
- **총 예상: ~$1.50/실험**

### 속도 제한
- API 요청 속도 제한 있음
- 과도한 요청 시 일시 차단 가능
- 실험 간 충분한 간격 유지 권장

### 재현성 한계
- 모델 업데이트로 인한 미세한 차이 가능
- 완전히 동일한 결과 보장 불가
- 통계적으로 유사한 결과 기대

## 🐛 문제 해결

### "API key not found" 에러
```bash
# .env 파일 확인
cat .env

# 환경변수 직접 설정
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

### "Rate limit exceeded" 에러
- 너무 많은 요청 → 몇 분 대기
- 토론 턴 수 줄이기: `max_turns=5`

### "Out of quota" 에러
- API 잔액 확인
- 결제 수단 등록 필요

### 메모리 부족
- 토큰 수 줄이기: `max_tokens=500`
- Docker 메모리 증가: `--memory=4g`

## 📞 지원

문제가 발생하면:
1. 이 가이드의 문제 해결 섹션 확인
2. GitHub Issues에 질문 등록
3. 로그 파일 첨부 (`outputs/.../full_transcript.json`)

## 📚 추가 자료

- [Anthropic API Docs](https://docs.anthropic.com)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Google Gemini Docs](https://ai.google.dev/docs)
- [논문 원문](링크)

## ✅ 체크리스트: 논문 제출 전

실험 재현성을 위해 다음을 확인하세요:

- [ ] 코드를 GitHub public repository에 업로드
- [ ] README.md에 설치/실행 방법 명시
- [ ] requirements.txt에 정확한 버전 명시
- [ ] 모든 시스템 프롬프트 공개
- [ ] 실험 설정 YAML/JSON 파일 포함
- [ ] 샘플 출력 파일 제공
- [ ] Docker 이미지 DockerHub에 업로드 (선택)
- [ ] 논문에 GitHub 링크 포함
- [ ] LICENSE 파일 추가
