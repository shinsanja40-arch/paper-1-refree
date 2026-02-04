# ---------------------------------------------------------------------------
# Dockerfile for Referee-Mediated Discourse Experiments
# ---------------------------------------------------------------------------
# Copyright (c) 2026 Cheongwon Choi <ccw1914@naver.com>
# Licensed under CC BY-NC 4.0
#   - Personal use allowed. Commercial use prohibited.
#   - Attribution required.
# ---------------------------------------------------------------------------

FROM python:3.10-slim

WORKDIR /app

# [FIX-CRITICAL-2] kiwipiepy C++ 빌드 의존성 추가
# kiwipiepy는 C++ 기반으로 컴파일이 필요하므로 build-essential 설치
# slim 이미지에는 컴파일러가 없어 빌드 실패하므로 사전 설치 필수
# [FIX-NEW-CRITICAL-3] apt 캐시 완전 정리로 이미지 크기 최적화
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# [FIX-CRITICAL-3] entrypoint.sh를 USER 변경 전에 복사하고 권한 설정
# USER appuser 후에는 root 소유 파일을 수정할 수 없으므로
# 사용자 생성 전에 모든 파일을 준비하고 chown으로 소유권 이전
COPY entrypoint.sh /app/entrypoint.sh
COPY referee_mediated_discourse.py /app/

# entrypoint.sh 실행 권한 부여 (root 권한으로)
RUN chmod +x /app/entrypoint.sh

# appuser 생성 및 /app 전체 소유권 이전
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app/outputs \
    && chown -R appuser:appuser /app

# 이제 appuser로 전환
USER appuser

# [FIX-CRITICAL-3] ENTRYPOINT만 설정하고 CMD는 제거
# 사용자가 docker run 또는 docker-compose에서 command로 인자 전달
ENTRYPOINT ["/app/entrypoint.sh"]

# CMD 제거 - 모든 인자는 command에서 받음
# 예: docker run image --debaters 4 --experiment nuclear_energy --seed 42
