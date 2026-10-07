# H2 EDU policy inference server.
#
# Deliberately slim: onnxruntime on CPU is enough for the single-environment,
# batch-1 graphs this serves (the ONNX exports declare a fixed batch of 1), and
# keeping Isaac Sim / torch out means this image rebuilds in seconds and is
# independent of both simulator containers.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WBC_H12_ROOT=/opt/wbc_h12 \
    H2_POLICY_BIND=tcp://0.0.0.0:5555

# numpy is pinned to the version gear_sonic pins (gear_sonic/pyproject.toml:21)
# so arrays crossing the wire behave identically on both sides.
RUN pip install --no-cache-dir \
        "numpy==1.26.4" \
        "onnxruntime==1.20.1" \
        "pyzmq>=25" \
        "joblib>=1.3"

WORKDIR /srv
COPY h2_policy_if/ /srv/h2_policy_if/
COPY server/ /srv/server/
COPY tools/ /srv/tools/

# No HEALTHCHECK curl: the server speaks ZeroMQ, not HTTP. tools/smoke_test.py
# is the equivalent and is run explicitly by run.sh.
ENTRYPOINT ["python", "server/h2_policy_server.py"]
