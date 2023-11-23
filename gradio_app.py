import gradio as gr
from gradio_model3dgs import Model3DGS
import os
from PIL import Image
import subprocess


try:
    from simple_knn._C import distCUDA2
except ImportError:
    os.system("pip install -e ./simple-knn")

try:
    from diff_gaussian_rasterization import GaussianRasterizer
except ImportError:
    os.system("pip install -e ./diff-gaussian-rasterization")


TMP_DIR = "/tmp"
os.makedirs(TMP_DIR, exist_ok=True)


def run(image_block: Image.Image):
    image_block.save(os.path.join(TMP_DIR, "tmp.png"))

    config_path = os.path.join("configs", "image.yaml")
    input_path = os.path.join(TMP_DIR, "tmp.png")
    subprocess.run(f"python train.py --config {config_path} input={input_path} save_path=tmp outdir={TMP_DIR}", shell=True)

    return os.path.join(TMP_DIR, "tmp_model.ply")


if __name__ == "__main__":
    title = "DreamGaussian Mini"

    example_folder = os.path.join(os.path.dirname(__file__), 'data')
    example_fns = os.listdir(example_folder)
    example_fns.sort()
    examples_full = [os.path.join(example_folder, x) for x in example_fns if x.endswith('.png')]

    with gr.Blocks(title=title) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("# " + title)
        
        with gr.Row(variant="panel"):
            with gr.Column(scale=5):
                image_block = gr.Image(type="pil", image_mode="RGBA", label="Input Image", height=300)
                gr.Examples(
                    examples=examples_full,
                    inputs=[image_block],
                    outputs=[image_block],
                    cache_examples=False,
                    label='Examples (click one of the images below to start)',
                    examples_per_page=40
                )
                img_run_btn = gr.Button("Generate 3D")
            
            with gr.Column(scale=5):
                model3dgs = Model3DGS(label="3DGS Model")
            
            img_run_btn.click(fn=run, inputs=[image_block], outputs=[model3dgs])

    demo.queue().launch(share=True)
