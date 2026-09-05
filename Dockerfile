FROM python:3.12-slim

WORKDIR /app

# requirements-render.txt pins the PyTorch CPU wheel index internally
# (--index-url), so torch installs as the CPU-only build (no ~2.7GB of unused
# CUDA libs) — do not add --extra-index-url here or pip will pull the CUDA wheel.
COPY requirements-render.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Single-vCPU free tier: cap thread pools so torch/OpenMP don't over-subscribe
# the core or inflate RSS with idle worker threads.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1

COPY poker_rl_agent/ poker_rl_agent/
COPY public/ public/
COPY configs/ configs/
# Deploy the certified Stage D 21k champion.
COPY checkpoints/snapshots/champion/stage_d_fchpa_21k.pt checkpoints/serving_champion.pt

RUN adduser --disabled-password --gecos '' appuser
USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "poker_rl_agent.serving.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
