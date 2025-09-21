import runpod
import io
import base64
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline

# Cargar la pipeline (desde la caché creada en build)
print("Cargando pipeline QwenImageEditPipeline...")
pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
pipeline.to("cuda")
print("Pipeline lista en GPU.")

# Funciones auxiliares
def b64_to_pil(b64_str):
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")

def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Handler en modo generador (para WebSocket streaming)
def generator_handler(event):
    """
    event = {
        "image": base64 (PNG/JPG),
        "mask": base64 (PNG),
        "prompt": str,
        "num_inference_steps": int (default 30),
        "true_cfg_scale": float (default 4.0),
        "seed": int (default 0),
        "negative_prompt": str (optional)
    }
    """
    image = b64_to_pil(event["image"])
    mask = b64_to_pil(event["mask"])
    prompt = event.get("prompt", "")
    steps = int(event.get("num_inference_steps", 30))
    cfg = float(event.get("true_cfg_scale", 4.0))
    seed = int(event.get("seed", 0))
    neg_prompt = event.get("negative_prompt", "")

    generator = torch.manual_seed(seed) if seed != 0 else None

    # Yield inicial
    yield {
        "status": "started",
        "progress": 0,
        "preview": None
    }

    # Callback para enviar previews cada cierto número de pasos
    def callback_fn(step, timestep, latents):
        percent = int(100 * (step / steps))
        with torch.no_grad():
            img_preview = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor).sample
            img_preview = (img_preview / 2 + 0.5).clamp(0, 1)
            preview = Image.fromarray(
                (img_preview[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            )
        yield {
            "status": "progress",
            "progress": percent,
            "preview": pil_to_b64(preview)
        }

    # Ejecutar pipeline con callback
    out = pipeline(
        image=image,
        prompt=prompt,
        mask_image=mask,
        num_inference_steps=steps,
        true_cfg_scale=cfg,
        generator=generator,
        negative_prompt=neg_prompt,
        callback=callback_fn,
        callback_steps=5   # cada 5 pasos
    )

    # Final
    result_img = out.images[0]
    yield {
        "status": "completed",
        "progress": 100,
        "preview": pil_to_b64(result_img)
    }

# Iniciar serverless con streaming
runpod.serverless.start({
    "handler": generator_handler,
    "return_aggregate_stream": True
})
