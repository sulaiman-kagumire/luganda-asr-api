FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi[standard] \
    uvicorn \
    git+https://github.com/nyrahealth/transformers.git@crisper_whisper \
    librosa \
    soundfile \
    torch \
    numpy \
    huggingface-hub \
    accelerate

RUN curl -o /usr/local/lib/python3.11/utils.py https://raw.githubusercontent.com/nyrahealth/CrisperWhisper/main/utils.py

COPY app.py .

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]