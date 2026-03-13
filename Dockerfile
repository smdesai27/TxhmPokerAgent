FROM python:3.12-slim

WORKDIR /app

COPY requirements-render.txt requirements.txt
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY poker_rl_agent/ poker_rl_agent/
COPY configs/ configs/
COPY checkpoints/latest.pt checkpoints/latest.pt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "poker_rl_agent.serving.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
