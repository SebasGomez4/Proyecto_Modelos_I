# Construir desde la RAÍZ del repositorio:  docker build -f fase-4/Dockerfile -t modelos-api-monitoreo:1.0 .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY fase-4/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código reutilizado de la Fase 2 (sin duplicarlo) y código de la API con registro
COPY fase-2/src/ src/
COPY fase-4/api.py fase-4/ajustes.py fase-4/entrada.py fase-4/entrenamiento.py fase-4/servicio_modelo.py fase-4/rutas.py fase-4/registro.py ./

# Datos, modelos y registro NO se copian: se montan al ejecutar (ver README). Nada se entrena en el build.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]