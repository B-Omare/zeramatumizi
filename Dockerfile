# ZeraMatumizi - Lightweight production container
# Runs FastAPI backend and Streamlit dashboard only
# Heavy ML training packages excluded (run training locally)

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY data/ data/
COPY docs/ docs/

# Install only what the API and dashboard need
RUN pip install --upgrade pip && \
    pip install -e . && \
    pip install \
    pandas==2.2.3 \
    pyarrow \
    fastapi \
    uvicorn \
    streamlit \
    plotly \
    xgboost \
    scikit-learn \
    groq \
    requests \
    python-dotenv \
    pydantic

# Startup script
RUN printf '#!/bin/bash\nuvicorn src.zeramatumizi.api.main:app --host 0.0.0.0 --port 8000 &\nstreamlit run src/zeramatumizi/dashboard/app.py --server.port 8501 --server.address 0.0.0.0\n' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 8000
EXPOSE 8501

CMD ["/bin/bash", "/app/start.sh"]