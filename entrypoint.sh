#!/bin/sh
# ---------------------------------------------------------------------------
# Entrypoint Script for Referee-Mediated Discourse Container
# ---------------------------------------------------------------------------
# Copyright (c) 2026 Cheongwon Choi <ccw1914@naver.com>
# Licensed under CC BY-NC 4.0
#   - Personal use allowed. Commercial use prohibited.
#   - Attribution required.
# ---------------------------------------------------------------------------
#
# [v5.10.0] Docker 권한 문제 완전 해결
# 
# 동작 흐름:
#   1. root로 실행됨 (Dockerfile에서 USER 설정 없음)
#   2. /app 및 /app/outputs 디렉토리 소유권 설정
#   3. 볼륨 마운트 권한 문제 해결 (chown)
#   4. gosu로 appuser로 전환하여 Python 실행
# ---------------------------------------------------------------------------

set -e

# 1. outputs 디렉토리 생성
mkdir -p /app/outputs

# 2. [CRITICAL-FIX] /app 디렉토리 소유권 설정 (탐색 권한 보장)
#    /app/outputs만 설정하면 상위 디렉토리 권한 문제 발생 가능
chown appuser:appuser /app

# 3. [CRITICAL] /app/outputs 소유권 변경
#    볼륨 마운트된 호스트 디렉토리도 appuser가 쓸 수 있도록 설정
chown -R appuser:appuser /app/outputs

# 4. gosu로 appuser 전환하여 Python 실행
exec gosu appuser python3 referee_mediated_discourse.py "$@"
