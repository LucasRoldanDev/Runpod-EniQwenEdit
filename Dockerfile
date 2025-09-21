FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Instalar dependencias básicas
RUN apt-get update && apt-get install -y \
    git wget curl python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Instalar librerías necesarias
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip3 install diffusers transformers accelerate safetensors pillow

# Pre-descargar el modelo Qwen-Image-Edit
RUN python3 - <<'PY'
from diffusers import QwenImageEditPipeline
print("Descargando modelo Qwen-Image-Edit...")
QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
print("✅ Descarga completada.")
PY

# Copiar tu handler
WORKDIR /app
COPY handler.py .

CMD ["python3", "-u", "handler.py"]
