
FROM python:3.11-slim

RUN python -m pip install --no-cache-dir \
    pytest==8.4.2 \
    pydantic==2.12.3

WORKDIR /workspace
