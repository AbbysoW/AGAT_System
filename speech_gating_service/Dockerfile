FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ \
    libsndfile1 \
    libasound2-dev \
    pulseaudio-utils \
    portaudio19-dev \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY client.py .
COPY speech_gating_service.py .
COPY logger/ ./logger/
COPY modules/ ./modules/

EXPOSE 8004



CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8004", "--reload"]