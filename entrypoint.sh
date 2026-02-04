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
# 존재 이유:
#   docker run -v ./outputs:/app/outputs 를 사용하면 호스트 폴더가
#   컨테이너 내부 /app/outputs를 덮어씁니다.
#   Dockerfile의 chown은 image build 시점에만 적용되어, 볼륨 마운트 후
#   호스트 폴더의 UID가 컨테이너 appuser와 다르면 Permission Denied가 발생합니다.
#
#   이 스크립트는 컨테이너가 실제로 시작될 때마다 outputs/ 권한을
#   재설정하여 appuser가 반드시 쓸 수 있도록 합니다.
#
# 동작:
#   1. /app/outputs가 없으면 생성
#   2. /app/outputs 소유권을 appuser로 재설정
#      [FIX-14] chmod 777 → chown appuser:appuser
#        777은 모든 사용자에게 읽기·쓰기·실행 권한을 부여하여
#        보안 취약점이 될 수 있습니다.
#        chown으로 소유권만 변경하면 appuser만 쓸 수 있어 더 안전합니다.
#   3. 전달받은 모든 인자를 python referee_mediated_discourse.py에 전달

set -e

mkdir -p /app/outputs
chown -R appuser:appuser /app/outputs

exec python referee_mediated_discourse.py "$@"
