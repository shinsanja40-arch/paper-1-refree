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

# [v5.10.0] gosu 설치 + 서명 검증 (보안 강화)
# gosu: 경량 권한 전환 도구 (su/sudo 대체)
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    wget \
    ca-certificates \
    gnupg \
    && GOSU_VERSION=1.19 \
    && ARCH="$(dpkg --print-architecture)" \
    && wget -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/${GOSU_VERSION}/gosu-${ARCH}" \
    && wget -O /tmp/gosu.asc "https://github.com/tianon/gosu/releases/download/${GOSU_VERSION}/gosu-${ARCH}.asc" \
    && export GNUPGHOME="$(mktemp -d)" \
    && gpg --batch --keyserver hkps://keys.openpgp.org --recv-keys B42F6819007F00F88E364FD4036A9C25BF357DD4 \
    && gpg --batch --verify /tmp/gosu.asc /usr/local/bin/gosu \
    && rm -rf "$GNUPGHOME" /tmp/gosu.asc \
    && chmod +x /usr/local/bin/gosu \
    && gosu --version \
    && gosu nobody true \
    && apt-get purge -y gnupg wget \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# entrypoint 및 메인 스크립트 복사
COPY entrypoint.sh /app/entrypoint.sh
COPY referee_mediated_discourse.py /app/

# entrypoint.sh 실행 권한 부여
RUN chmod +x /app/entrypoint.sh

# appuser 생성 및 /app 소유권 설정
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app/outputs \
    && chown -R appuser:appuser /app

# [v5.10.0] USER 설정 제거 (gosu로 안전한 권한 전환)
# entrypoint.sh가 root로 실행되어 gosu로 전환

# ENTRYPOINT는 root로 실행됨
ENTRYPOINT ["/app/entrypoint.sh"]
