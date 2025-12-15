FROM python:3.12-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ musl-dev curl libffi-dev gfortran libopenblas-dev \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Create the /app/py directory
RUN mkdir -p /app/py
WORKDIR /app/py
COPY pyproject.toml ./
RUN pip install --upgrade uv
RUN uv pip install --system -e ".[core]" -e ".[opentelemetry]" && \
    uv pip install --system gunicorn uvicorn pydantic

# Create the final image
FROM python:3.12-slim

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the built environment from builder to final image
# (If you want a fully self-contained environment, copy /usr/local)
COPY --from=builder /usr/local /usr/local

WORKDIR /app

# Copy the rest of your source code
COPY . /app

# Expose environment variables and port
ARG R2R_PORT=8000 R2R_HOST=0.0.0.0
ENV R2R_PORT=$R2R_PORT R2R_HOST=$R2R_HOST
EXPOSE $R2R_PORT

# Launch the app with OpenTelemetry auto-instrumentation
# The opentelemetry-instrument CLI wrapper automatically instruments FastAPI,
# database clients, HTTP requests, and generates standard HTTP metrics
CMD ["sh", "-c", "opentelemetry-instrument uvicorn core.main.app_entry:app --host $R2R_HOST --port $R2R_PORT"]
