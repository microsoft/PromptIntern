FROM python:3.12-slim

RUN useradd --create-home --uid 10001 runner
COPY worker.py /opt/worker.py

USER 10001:10001
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "/opt/worker.py"]