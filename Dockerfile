FROM python:3.10-slim

WORKDIR /app

# System dependencies — git는 미사용하여 제거

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY referee_mediated_discourse.py .
COPY README.md .

# outputs/ 디렉토리를 비루트 사용자 소유로 생성
# docker-compose 볼륨 마운트 시 로컬 폴더 권한과 충돌하지 않도록
# appuser로 소유권을 지정합니다.
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app/outputs \
    && chown -R appuser:appuser /app

USER appuser

# API 키는 빈 문자열로 기본 설정 (runtime에 오버라이드됨)
ENV ANTHROPIC_API_KEY=""
ENV OPENAI_API_KEY=""
ENV GOOGLE_API_KEY=""

# entrypoint.sh: 볼륨 마운트 후 outputs/ 권한 재설정 → python 실행
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ENTRYPOINT: --debaters 기본값 4 포함
ENTRYPOINT ["/app/entrypoint.sh", "--debaters", "4"]

# CMD: 기본 실험 설정 (ENTRYPOINT 뒤에 추가됨)
CMD ["--experiment", "nuclear_energy", "--seed", "42"]
