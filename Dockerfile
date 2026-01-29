FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY referee_mediated_discourse.py .
COPY README.md .

# Create output directory
RUN mkdir -p outputs

# Set environment variables (will be overridden at runtime)
ENV ANTHROPIC_API_KEY=""
ENV OPENAI_API_KEY=""
ENV GOOGLE_API_KEY=""

# Default command
ENTRYPOINT ["python", "referee_mediated_discourse.py"]
CMD ["--experiment", "nuclear_energy", "--seed", "42"]
