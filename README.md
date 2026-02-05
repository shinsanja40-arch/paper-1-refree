# Referee-Mediated Discourse: Reproducible Experimental Protocol

**Version 5.14.0 FINAL** - Production Ready

Multi-agent debate framework with real-time hallucination detection and correction.

## 🎯 Overview

- Multi-agent adversarial debates (4명 이상의 토론자)
- Real-time hallucination detection via independent referee (Gemini)
- Turn-by-turn error correction with per-turn timeout enforcement
- Comprehensive logging and ML-ready evaluation output
- Standardized metrics calculation
- **[v5.14.0]** --timeout 명령행 인자 추가 (사용자 정의 timeout 지원)
- **[v5.14.0]** .env.example 보안 주의사항 강화
- **[v5.13.0]** seed 검증 완전 강화 (1 ~ 2^31-1, 음수 명시적 거부)
- **[v5.12.0]** kiwi.tokenize() lock 추가 (완전한 thread-safety 보장)
- **[v5.12.0]** 외부 AI 검증 완료 (7개 지적사항, 1개 실제 버그 수정)
- **[v5.11.0]** seed 검증 완전 적용 + 타임스탬프 밀리초 적용

## 📋 Prerequisites

**[v5.3.0 Important]** 이 버전은 **새로운 Google Gemini SDK**(`google-genai`)를 사용합니다.  
기존 버전에서 업그레이드하는 경우:
```bash
pip uninstall google-generativeai
pip install google-genai
```

## 📋 System Requirements

- Python 3.10+
- API keys: Anthropic (Claude), OpenAI (GPT-4o), Google (Gemini)

## 🚀 Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd referee-mediated-discourse

# [중요] outputs 디렉토리 사전 생성 (권한 문제 방지)
mkdir -p outputs

pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# .env를 열어 실제 API 키 입력
```

Or export directly:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
```

### 3. Run Experiments

```bash
# 원자력 토론 (4명 토론자, 기본값)
python referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 42

# 철학 토론
python referee_mediated_discourse.py --experiment good_vs_evil --debaters 4 --seed 42

# 6명 토론자 확장 실험
python referee_mediated_discourse.py --experiment nuclear_energy --debaters 6 --seed 42

# 사용자 정의 출력 디렉토리
python referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 42 --output-dir ./my_results

# 사용자 정의 timeout (고성능 모델 대응)
python referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 42 --timeout 120
```

### 4. Docker

**[v5.8.0 Important]** Docker 실행 환경 완전 최적화:
- `gosu` 도구로 안전한 권한 전환 (python:3.10-slim 호환)
- entrypoint가 root로 볼륨 권한 자동 수정
- 이후 자동으로 appuser로 전환하여 실행
- 모든 호스트 환경에서 안정적 동작 보장

#### Docker 빌드 및 실행 (완전 가이드)

```bash
# 1. outputs 폴더 사전 생성 (권한 문제 방지)
mkdir -p outputs

# 2. 이미지 빌드
docker build -t referee-debate:latest .

# 3-A. 기본 실행 (환경변수 직접 전달)
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e OPENAI_API_KEY="sk-..." \
  -e GOOGLE_API_KEY="AIza..." \
  referee-debate:latest \
  --debaters 4 --experiment nuclear_energy --seed 42

# 3-B. .env 파일 사용 (권장 방법)
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  --env-file .env \
  referee-debate:latest \
  --debaters 4 --experiment nuclear_energy --seed 42

# 4. 6명 토론자 실험
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  --env-file .env \
  referee-debate:latest \
  --debaters 6 --experiment nuclear_energy --seed 99

# 5. 사용자 정의 seed
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  --env-file .env \
  referee-debate:latest \
  --debaters 4 --experiment good_vs_evil --seed 123
```

#### Docker Compose 사용 (완전 가이드)

```bash
# 1. outputs 폴더 사전 생성
mkdir -p outputs

# 2. 기본 실험 (referee-debate 서비스)
docker compose up referee-debate

# 3. [중요] Profiles 사용 방법
# 철학 토론 (philosophy profile)
docker compose --profile philosophy up philosophy-debate

# 6명 토론자 (extended profile)
docker compose --profile extended up six-debaters

# 또는 service 이름으로 직접 실행 (profile 자동 활성화)
docker compose up philosophy-debate  # --profile 생략 가능
docker compose up six-debaters       # --profile 생략 가능

# 백그라운드 실행
docker compose up -d referee-debate

# 로그 확인
docker compose logs -f referee-debate

# 정지 및 삭제
docker compose down
```

## 📊 Output Structure

```
outputs/
└── nuclear_energy_4d_2025-01-29T10-30-45/
    ├── config.json                      # 실험 설정 전체
    ├── full_transcript.json             # 턴별 대화 로그
    ├── referee_decisions.json           # 심판 판결 이력
    ├── hallucination_annotations.json   # 환각 탐지 결과
    └── metrics.json                     # 정량 지표
```

## 🔬 Architecture

```
┌─────────────┐         ┌─────────────┐
│  Debater 1  │◄───────►│  Debater 2  │
│  (Claude)   │         │  (GPT-4o)   │
└──────┬──────┘         └──────┬──────┘
       │                       │
┌──────┴──────┐         ┌──────┴──────┐
│  Debater 3  │         │  Debater 4  │
│  (Claude)   │         │  (GPT-4o)   │
└──────┬──────┘         └──────┬──────┘
       │         ┌─────────────┘
       └────────►│   Referee   │  ← stateless, per-turn timeout
                 │  (Gemini)   │
                 └─────────────┘
```

**토론 흐름:**
1. 각 토론자가 라운드별로 차례 발언 (동료 발언 포함)
2. 심판이 라운드 종료 후 모든 발언을 사실 검증
3. 교착 탐지(Jaccard 유사도) → 반복 시 자동 종료
4. 각 호출에 스레드 타임아웃 적용 → 무한 대기 방지

## 📈 Metrics

| Metric | Description |
|--------|-------------|
| hallucination_rate | 토론자 턴 중 환각 포함 비율 |
| correction_rate | 탐지된 환각 중 수정 제공 비율 |
| factual_errors | 사실 오류 건수 |
| unverifiable_claims | 검증 불가 주장 건수 |
| misleading_claims | 오도적 주장 건수 |

## 🔄 Reproducibility

```bash
# 동일한 seed로 재실행하면 동일한 실험 구성
python referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 42

# 다른 seed로 실험하여 재현성 테스트
python referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 123
python referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed 999
```

- Fixed random seeds
- Pinned dependencies (requirements.txt)
- Complete config logging per run
- Timestamped outputs (덮어쓰기 없음)

## 🛠️ Customization

### 새로운 실험 추가

```python
# 1. _create_balanced_debaters() 내 topic_a / topic_b 조정
# 2. main()의 --experiment choices에 추가
# 3. 초기 프롬프트 조정
```

### 토론자 수 조정

```bash
# 반드시 >= 4 이고 짝수여야 함
--debaters 4   # 기본: Strong A, Moderate A, Strong B, Moderate B
--debaters 6   # 확장: Neutral x2 추가
--debaters 8   # 각 스턴스 x2
```

## 🛠 Troubleshooting

### Docker 실행 예시 (상세)

```bash
# 이미지 빌드
docker build -t referee-debate:latest .

# 기본 실행 (환경변수 직접 전달)
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e OPENAI_API_KEY="sk-..." \
  -e GOOGLE_API_KEY="AIza..." \
  referee-debate:latest \
  --debaters 4 --experiment nuclear_energy --seed 42

# .env 파일 사용 (권장)
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  --env-file .env \
  referee-debate:latest \
  --debaters 4 --experiment nuclear_energy --seed 42
```

### Docker 관련 FAQ

**Q: Docker Compose에서 서비스가 실행 안 됨**
```bash
# Profile을 명시해야 합니다
docker compose --profile extended up six-debaters

# 또는 service 이름으로 직접 실행 (profile 자동 활성화)
docker compose up six-debaters
```

**Q: 권한 오류 발생**
```bash
# outputs 폴더 사전 생성
mkdir -p outputs

# 또는 Docker가 자동으로 생성하고 gosu가 권한 수정
```

**Q: Apple Silicon (M1/M2/M3)에서 빌드 실패**
```bash
# Dockerfile의 gosu가 아키텍처 자동 감지
# 수동 빌드 시:
docker build --platform linux/arm64 -t referee-debate:latest .
```

### 일반적인 문제

| 증상 | 해결 |
|------|------|
| `ValueError: ...API_KEY not set` | `.env` 파일 또는 환경변수 확인 |
| `Permission Denied` (Docker) | `mkdir -p outputs` 후 재실행 |
| Rate limit exceeded | `--debaters 4`로 줄이거나 잠시 대기 |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| 무한 대기 | turn_timeout(60s)이 적용됨 — 자동 복구 |
| Docker: `--debaters` 무시됨 | command에서 파라미터 전달 (위 예시 참고) |
| Gemini JSON 파싱 오류 | 자동 재시도됨, 로그에서 상세 확인 |

## 📝 Citation

```bibtex
@article{referee_mediated_discourse_2026,
  title={Breaking the Data Wall: High-Fidelity Knowledge Synthesis
         and Self-Evolving AI via Referee-Mediated Discourse},
  author={Cheongwon Choi},
  year={2026}
}
```

## 📚 Further Reading

- [Anthropic Claude Docs](https://docs.anthropic.com)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Google Gemini API Guide](https://ai.google.dev/docs)

## 📄 License

Copyright (c) 2026 Cheongwon Choi <ccw1914@naver.com>

Licensed under CC BY-NC 4.0:
- ✅ Personal use allowed
- ❌ Commercial use prohibited
- ✅ Attribution required
- Full terms: https://creativecommons.org/licenses/by-nc/4.0/
