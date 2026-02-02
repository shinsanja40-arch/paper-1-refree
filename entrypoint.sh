#!/bin/sh
# entrypoint.sh — 컨테이너 시작 시 실행
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
#   2. /app/outputs 권한을 777로 설정 (컨테이너 내부 프로세스용)
#   3. 전달받은 모든 인자를 python referee_mediated_discourse.py에 전달

set -e

mkdir -p /app/outputs
chmod 777 /app/outputs

exec python referee_mediated_discourse.py "$@"
