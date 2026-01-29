#!/bin/bash

# Quick Start Script for Referee-Mediated Discourse Experiments
# This script helps you get started quickly

set -e

echo "=================================================="
echo "Referee-Mediated Discourse - Quick Start"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
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
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

echo ""
echo "🔑 Checking API keys..."

# Check for .env file
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

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check if keys are set
missing_keys=0
if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your_anthropic_api_key_here" ]; then
    echo "❌ ANTHROPIC_API_KEY not set in .env"
    missing_keys=1
fi

if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "❌ OPENAI_API_KEY not set in .env"
    missing_keys=1
fi

if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "your_google_api_key_here" ]; then
    echo "❌ GOOGLE_API_KEY not set in .env"
    missing_keys=1
fi

if [ $missing_keys -eq 1 ]; then
    echo ""
    echo "Please edit .env and add your API keys, then run this script again."
    exit 1
fi

echo "✅ All API keys configured"
echo ""

# Create outputs directory
mkdir -p outputs

echo "=================================================="
echo "🚀 Ready to run experiments!"
echo "=================================================="
echo ""
echo "Available experiments:"
echo "  1. Nuclear Energy Debate"
echo "  2. Good vs Evil Philosophical Debate"
echo ""
echo "Choose an experiment (1 or 2): "
read -r choice

case $choice in
    1)
        echo ""
        echo "🔬 Running Nuclear Energy Debate..."
        python referee_mediated_discourse.py --experiment nuclear_energy --seed 42
        ;;
    2)
        echo ""
        echo "🔬 Running Good vs Evil Debate..."
        python referee_mediated_discourse.py --experiment good_vs_evil --seed 42
        ;;
    *)
        echo "Invalid choice. Please run the script again and choose 1 or 2."
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "✅ Experiment completed!"
echo "=================================================="
echo ""
echo "📁 Results are saved in the outputs/ directory"
echo ""
echo "To run another experiment, execute:"
echo "  python referee_mediated_discourse.py --experiment [nuclear_energy|good_vs_evil] --seed 42"
echo ""
echo "To deactivate the virtual environment, run:"
echo "  deactivate"
echo ""
