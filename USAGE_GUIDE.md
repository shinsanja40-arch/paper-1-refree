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
├── USAGE_GUIDE.md                 # 이 파일 (한글 가이드)
├── quickstart.sh                  # 빠른 시작 스크립트
├── Dockerfile                     # Docker 컨테이너 설정
├── docker-compose.yml             # Docker Compose 설정
└── .gitignore                     # Git 무시 파일 목록
```

## 🚀 빠른 시작 (3가지 방법)

### 방법 1: 자동 스크립트 사용 (가장 쉬움)

```bash
chmod +x quickstart.sh
./quickstart.sh
# 화면의 지시를 따라 API 키 설정 후 실험 선택 (1, 2, 3)
# seed 값을 직접 입력하거나 Enter로 기본값(42) 사용
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

# 4. 실험 실행 (--debaters는 반드시 >= 4이고 짝수)
python3 referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 42
```

### 방법 3: Docker 사용 (환경 격리)

```bash
# 1. outputs/ 폴더 사전 생성 (볼륨 권한 문제 방지)
mkdir -p outputs

# 2. 이미지 빌드
docker build -t referee-debate .

# 3. 4명 토론자 실험 (중요: 모든 파라미터를 command로 전달)
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  -e ANTHROPIC_API_KEY="your-key" \
  -e OPENAI_API_KEY="your-key" \
  -e GOOGLE_API_KEY="your-key" \
  referee-debate \
  --debaters 4 --experiment nuclear_energy --seed 42

# 4. 6명 토론자 실험
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  -e ANTHROPIC_API_KEY="your-key" \
  -e OPENAI_API_KEY="your-key" \
  -e GOOGLE_API_KEY="your-key" \
  referee-debate \
  --debaters 6 --experiment nuclear_energy --seed 99

# 5. 사용자 정의 seed로 재현성 테스트
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  -e ANTHROPIC_API_KEY="your-key" \
  -e OPENAI_API_KEY="your-key" \
  -e GOOGLE_API_KEY="your-key" \
  referee-debate \
  --debaters 4 --experiment good_vs_evil --seed 123

# 6. Docker Compose 사용 (.env 파일 자동 로드)
mkdir -p outputs
docker compose up referee-debate

# 철학 토론
docker compose --profile philosophy up philosophy-debate

# 6명 토론자
docker compose --profile extended up six-debaters
```

**중요**: Docker 실행 시 `--debaters`, `--experiment`, `--seed`는 반드시 command에서 전달해야 합니다.

## 🔑 API 키 발급 방법

### 1. Anthropic Claude API
- https://console.anthropic.com 접속 → API Keys → 키 생성
- 비용: 종량제

### 2. OpenAI GPT API
- https://platform.openai.com 접속 → API keys → 키 생성
- 비용: 종량제

### 3. Google Gemini API
- https://ai.google.dev 접속 → 프로젝트 생성 → API key 생성
- 비용: 무료 티어 있음 (제한적)

## 🧪 실험 실행

### 명령어 형식

```bash
python3 referee_mediated_discourse.py \
  --experiment [nuclear_energy|good_vs_evil] \
  --debaters  [4|6|8|...]          # >= 4, 짝수만 가능
  --seed      [난수 시드]          # 재현성을 위한 시드 값
  --output-dir [출력 디렉토리]      # 기본값: outputs/
```

### 예시

```bash
# 원자력 에너지 토론 — 4명 토론자
python3 referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 42

# 철학 토론 — 4명 토론자, 다른 시드
python3 referee_mediated_discourse.py --experiment good_vs_evil --debaters 4 --seed 123

# 원자력 토론 — 6명 토론자 (Neutral Analyst x2 추가)
python3 referee_mediated_discourse.py --experiment nuclear_energy --debaters 6 --seed 42

# 재현성 테스트 — 다른 seed로 여러 번 실행
python3 referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 42
python3 referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 123
python3 referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 999

# 사용자 정의 출력 디렉토리
python3 referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 42 --output-dir ./my_results
```

### --debaters 옵션 설명

| 값 | 토론자 구성 |
|----|------------|
| 4 | Strong A, Moderate A, Strong B, Moderate B |
| 6 | 위 4명 + Neutral Analyst x2 |
| 8 이상 | 각 스턴스를 균등 배분 |

### --seed 옵션 설명

- **목적**: 실험의 완전한 재현성 보장
- **기본값**: 42
- **사용 예**:
  - 동일한 seed → 동일한 토론자 배치 및 초기 상태
  - 다른 seed → 새로운 실험 조건으로 변동성 테스트
- **재현성 테스트**: 동일 실험을 서로 다른 seed로 여러 번 실행하여 결과 안정성 확인

## 📊 출력 파일 설명

```
outputs/nuclear_energy_4d_2025-01-29T10-30-45/
├── config.json                      # 실험 설정 전체
├── full_transcript.json             # 턴별 대화 로그 (논리 단계 포함)
├── referee_decisions.json           # 심판 판결 이력
├── hallucination_annotations.json   # 환각 탐지 및 수정 기록
└── metrics.json                     # 정량적 지표
```

### config.json
실험의 모든 파라미터를 기록합니다. 재현을 위해 필요한 모든 정보가 포함됩니다.

### full_transcript.json
```json
[
  {
    "round_number": 1,
    "turn_number": 1,
    "agent_role": "debater_1",
    "agent_name": "Strong Nuclear Energy Advocate",
    "model": "claude-3-5-sonnet-20241022",
    "content": "...",
    "tokens_used": 450,
    "latency_ms": 1234.56,
    "metadata": {
      "fallback": false,
      "skip_reason": null
    }
  }
]
```

### referee_decisions.json
```json
[
  {
    "round_number": 1,
    "turn_number": 1,
    "target_debater": "Strong Nuclear Energy Advocate",
    "claim": "원자력 발전소의 사고율은 0.1%...",
    "decision": "FACTUAL_ERROR",
    "reasoning": "실제 사고율은...",
    "evidence": ["https://..."],
    "correction": "올바른 수치는..."
  }
]
```

### metrics.json
```json
{
  "total_turns": 25,
  "debater_turns": 20,
  "referee_interventions": 5,
  "hallucination_rate": 0.15,
  "correction_rate": 0.80,
  "factual_errors": 2,
  "unverifiable_claims": 1,
  "misleading_claims": 0,
  "corrections_provided": 2
}
```

## 🔬 논문 재현을 위한 체크리스트

- [ ] **동일한 시드 사용**: `--seed 42`
- [ ] **동일한 토론자 수**: `--debaters 4`
- [ ] **동일한 모델 버전**: config.json에서 확인
- [ ] **metrics.json 비교**: hallucination_rate, correction_rate
- [ ] **full_transcript.json 검토**: 실제 대화 내용 확인
- [ ] **다중 seed 테스트**: 재현성 검증을 위해 다른 seed로도 실행

## 🛠️ 커스터마이징

### 새로운 실험 추가

`referee_mediated_discourse.py`를 수정합니다:

```python
# 1. _create_balanced_debaters()에서 topic_a, topic_b 조정
if experiment_name == "climate_debate":
    topic_a, topic_b = "aggressive action", "gradual transition"

# 2. main()의 argparse choices에 추가
choices=["nuclear_energy", "good_vs_evil", "climate_debate"]
```

### 파라미터 조정

```python
# ExperimentConfig 내부 값 조정 가능
max_rounds=5           # 토론 라운드 수 (기본값: 5)
turn_timeout=60        # 턴당 타임아웃 초 (기본값: 60)
deadlock_threshold=3   # 교착 판정 반복 횟수 (기본값: 3)
max_context_turns=10   # 컨텍스트 윈도우 크기 (기본값: 10)
```

## 📈 결과 분석

### Python에서 결과 로드

```python
import json

with open('outputs/.../metrics.json') as f:
    metrics = json.load(f)

print(f"환각률: {metrics['hallucination_rate']:.2%}")
print(f"교정률: {metrics['correction_rate']:.2%}")

with open('outputs/.../full_transcript.json') as f:
    transcript = json.load(f)

for turn in transcript[:5]:
    print(f"\n=== Round {turn['round_number']} — {turn['agent_name']} ===")
    print(turn['content'][:200])
```

### 여러 실험 비교

```bash
# 다른 시드로 3번 실행하여 재현성 확인
python3 referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 42
python3 referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 123
python3 referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 999

# 결과 비교 스크립트 예시
python3 -c "
import json, glob
for path in glob.glob('outputs/nuclear_energy_4d_*/metrics.json'):
    with open(path) as f:
        m = json.load(f)
    print(f'{path}: {m[\"hallucination_rate\"]:.2%}')
"
```

## ⚠️ 주의사항

### API 비용 (예상)
| 구성 | 비용 |
|------|------|
| Agent (Claude, 4턴) | ~$0.50 |
| Agent (GPT-4o, 4턴) | ~$1.00 |
| Referee (Gemini) | 무료 (한도 내) |
| **4명 토론자 합계** | **~$3.00/실험** |

### 재현성 한계
- 모델 업데이트로 인한 미세한 차이 가능
- 완전히 동일한 결과 보장 불가
- 통계적으로 유사한 결과 기대
- 동일 seed로 여러 번 실행 시 결과 비교

## 🐛 문제 해결

| 증상 | 해결 방법 |
|------|-----------|
| `API key not found` | `.env` 파일 또는 환경변수 확인 |
| `Permission Denied` (Docker) | `mkdir -p outputs` 후 재실행 |
| Rate limit exceeded | 잠시 대기 후 재실행 |
| 무한 대기 | turn_timeout(60s)이 자동 적용됨 |
| `--debaters` 에러 | 값이 >= 4 이고 짝수인지 확인 |
| Docker에서 `--debaters` 무시 | command에서 파라미터 전달 (위 예시 참고) |
| Gemini JSON 파싱 실패 | 로그 파일에서 상세 오류 확인, 자동 재시도됨 |
| Gemini 모델 초기화 실패 | API 키 확인, 모델 이름 확인 (gemini-1.5-pro) |
| FileNotFoundError | output 디렉토리 자동 생성됨, 권한 확인 |

## ✅ 체크리스트: 논문 제출 전

- [ ] 코드를 GitHub public repository에 업로드
- [ ] README.md에 설치/실행 방법 명시
- [ ] requirements.txt에 정확한 버전 명시
- [ ] 모든 시스템 프롬프트 공개
- [ ] Docker 이미지 빌드 및 테스트 완료
- [ ] LICENSE 파일 추가
- [ ] 재현성 테스트 (동일 seed + 다른 seed 모두 실행)
- [ ] 문서에 Docker 실행 방법 정확히 기술

## 📄 라이센스

Copyright (c) 2026 Cheongwon Choi <ccw1914@naver.com>

Licensed under CC BY-NC 4.0:
- ✅ 개인 사용 허용
- ❌ 상업적 사용 금지
- ✅ 저작자 표시 필수
- 전체 약관: https://creativecommons.org/licenses/by-nc/4.0/
