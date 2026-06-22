FROM python:3.12-slim

WORKDIR /app

COPY requirements-render.txt requirements.txt
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY poker_rl_agent/ poker_rl_agent/
COPY public/ public/
COPY configs/ configs/
# Deploy the certified Stage D 21k champion (not the gate-failing latest.pt).
COPY checkpoints/snapshots/interview_ready/interview_ready_1.pt checkpoints/serving_champion.pt

RUN adduser --disabled-password --gecos '' appuser
USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "poker_rl_agent.serving.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
