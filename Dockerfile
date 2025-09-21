# Imagen base con CUDA y Python
FROM runpod/base:0.4.0-cuda11.8.0

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descargar modelo Qwen-Image-Edit
RUN python - <<'PY'
from diffusers import QwenImageEditPipeline
print("Descargando modelo Qwen-Image-Edit...")
QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
print("Descarga completada.")
PY

# Copiar handler
COPY handler.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
