#!/bin/bash
# ---------------------------------------------------------------------------
# Quick Start Script for Referee-Mediated Discourse Experiments
# 실험 시작을 위한 빠른 설정 스크립트
#
# Copyright (c) 2026 Cheongwon Choi <ccw1914@naver.com>
# Licensed under CC BY-NC 4.0
#   - Personal use allowed.  Commercial use prohibited.
#   - Attribution required.
# ---------------------------------------------------------------------------

set -e

echo "=================================================="
echo "Referee-Mediated Discourse - Quick Start"
echo "=================================================="
echo ""

# ── Python 확인 ──────────────────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi
echo "✅ Python found: $(python3 --version)"
echo ""

# ── 가상환경 ───────────────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip
# [FIX-NEW-CRITICAL-2] requirements.txt 오타 수정 확인
pip install -r requirements.txt
echo "✅ Dependencies installed"

# ── API 키 확인 ─────────────────────────────────────────────────────────────
echo ""
echo "🔑 Checking API keys..."

if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating from template..."
    cp .env.example .env
    echo ""
    echo "📝 Please edit .env and add your API keys:"
    echo "   - ANTHROPIC_API_KEY"
    echo "   - OPENAI_API_KEY"
    echo "   - GOOGLE_API_KEY"
    echo ""
    echo "Then run this script again."
    exit 0
fi

# .env에서 환경변수 로드
# set -a : 이후 source된 변수를 자동으로 export
# xargs 방식은 키 값에 공백·특수문자가 포함되면 word-splitting으로 오작동.
set -a
source .env
set +a

# [FIX-MEDIUM-P2] API 키 검증 강화 (Gemini 제안)
missing_keys=0
validate_key() {
    local key_name=$1
    local key_value=$2
    
    if [ -z "$key_value" ] ||        [ "$key_value" = "your_${key_name,,}_here" ] ||        [ "$key_value" = "" ] ||        [[ "$key_value" =~ ^your_ ]]; then
        echo "❌ $key_name not set properly in .env"
        return 1
    fi
    return 0
}

validate_key "ANTHROPIC_API_KEY" "$ANTHROPIC_API_KEY" || missing_keys=1
validate_key "OPENAI_API_KEY" "$OPENAI_API_KEY" || missing_keys=1
validate_key "GOOGLE_API_KEY" "$GOOGLE_API_KEY" || missing_keys=1

if [ $missing_keys -eq 1 ]; then
    echo ""
    echo "Please edit .env and add your API keys, then run this script again."
    exit 1
fi

echo "✅ All API keys configured"
echo ""

# ── outputs/ 폴더 생성 (현재 사용자 소유로) ─────────────────────────────────
# Docker 볼륨 마운트 시 root 소유 폴더가 생기지 않도록 사전 생성합니다.
mkdir -p outputs
echo "✅ outputs/ directory ready"

# ── 실험 및 seed 선택 ───────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "🚀 Ready to run experiments!"
echo "=================================================="
echo ""
echo "Available experiments:"
echo "  1. Nuclear Energy Debate (4명 토론자)"
echo "  2. Good vs Evil Philosophical Debate (4명 토론자)"
echo "  3. Nuclear Energy — 6명 토론자 (확장)"
echo ""
echo "Choose an experiment (1, 2, or 3): "
read -r choice

# [FIX-13] seed를 사용자 입력으로 받습니다.
#   기존: 항상 42로 고정 → 재현성 테스트 어려움
#   수정: 빈 값이면 기본값 42 사용, 숫자 아니면 오류 처리
echo ""
echo "Enter random seed (default: 42): "
read -r seed_input

if [ -z "$seed_input" ]; then
    SEED=42
else
    # 숫자 여부 검증
    if ! [[ "$seed_input" =~ ^[0-9]+$ ]]; then
        echo "⚠️  Invalid seed value '$seed_input'. Using default seed 42."
        SEED=42
    else
        SEED=$seed_input
    fi
fi
echo "🎲 Using seed: $SEED"

case $choice in
    1)
        echo ""
        echo "🔬 Running Nuclear Energy Debate (4 debaters, seed=$SEED)..."
        python3 referee_mediated_discourse.py \
            --experiment nuclear_energy --debaters 4 --seed "$SEED"
        ;;
    2)
        echo ""
        echo "🔬 Running Good vs Evil Debate (4 debaters, seed=$SEED)..."
        python3 referee_mediated_discourse.py \
            --experiment good_vs_evil --debaters 4 --seed "$SEED"
        ;;
    3)
        echo ""
        echo "🔬 Running Nuclear Energy Debate (6 debaters, seed=$SEED)..."
        python3 referee_mediated_discourse.py \
            --experiment nuclear_energy --debaters 6 --seed "$SEED"
        ;;
    *)
        echo "Invalid choice. Please run the script again and choose 1, 2, or 3."
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "✅ Experiment completed!"
echo "=================================================="
echo ""
echo "📁 Results are saved in the outputs/ directory"
echo "📝 Detailed log: outputs/<experiment_dir>/debate.log"
echo ""
echo "To run another experiment manually (examples):"
echo "  python3 referee_mediated_discourse.py --experiment nuclear_energy --debaters 4 --seed $SEED"
echo "  python3 referee_mediated_discourse.py --experiment nuclear_energy --debaters 6 --seed 99"
echo "  python3 referee_mediated_discourse.py --experiment good_vs_evil --debaters 4 --seed 123"
echo ""
echo "To deactivate the virtual environment:"
echo "  deactivate"
echo ""
