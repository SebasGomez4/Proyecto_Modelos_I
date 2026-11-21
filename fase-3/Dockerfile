# Construir desde la RAÍZ del repositorio:  docker build -f fase-3/Dockerfile -t modelos-api:1.0 .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY fase-3/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código reutilizado de la Fase 2 (sin duplicarlo) y código de la API
COPY fase-2/src/ src/
COPY fase-3/api.py fase-3/ajustes.py fase-3/entrada.py fase-3/entrenamiento.py fase-3/servicio_modelo.py fase-3/rutas.py ./

# Datos y modelos NO se copian: se montan al ejecutar (ver README). El modelo no se entrena en el build.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]