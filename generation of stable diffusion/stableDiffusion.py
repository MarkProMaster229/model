from diffusers import StableDiffusionPipeline
import torch
import os
from datetime import datetime

def generate_image(prompt: str, output_dir: str = "output_dir"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_path = "model/path"#смотри внутри модели - то где будут веса

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe = pipe.to(device)

    image = pipe(prompt, num_inference_steps=100).images[0]

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"generated_{timestamp}.png")
    image.save(filename)

    print(f"Изображение сохранено {filename}")

if __name__ == "__main__":
    prompt = "пусть_тут_будет_нечто_ответственное"
    generate_image(prompt)
