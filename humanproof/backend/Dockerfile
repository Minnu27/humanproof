# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HP_ENV=prod
RUN groupadd -r app && useradd -r -g app -d /app -s /usr/sbin/nologin app \
 && mkdir -p /var/lib/humanproof && chown app:app /var/lib/humanproof
COPY --from=build /install /usr/local
WORKDIR /app
COPY humanproof ./humanproof
COPY tools ./tools
COPY models ./models
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
# Behind a TLS-terminating proxy: trust forwarded headers only from it.
CMD ["uvicorn", "humanproof.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "10.0.0.0/8,172.16.0.0/12", "--no-server-header", "--workers", "1"]
