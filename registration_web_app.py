from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from manuscript_registration.inference import (
    align_pil_images,
    alignment_overlay,
    flow_visualization,
    ink_bbox_affine_prealign,
    load_registration_model,
    tensor_to_pil,
)


DEFAULT_CHECKPOINT = Path(
    os.environ.get("REGISTRATION_CHECKPOINT", "models/manuscript-registration-best.pt")
)


@lru_cache(maxsize=2)
def cached_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_registration_model(checkpoint_path, device=device)


def run_alignment(source: Image.Image, target: Image.Image, checkpoint_path: str, maximum_width: int):
    if source is None or target is None:
        raise gr.Error("Please upload both source and target line images.")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise gr.Error(f"Checkpoint not found: {checkpoint}")
    model, checkpoint_data = cached_model(str(checkpoint.resolve()))
    output, source_tensor, target_tensor = align_pil_images(
        model,
        source,
        target,
        training_image_size=tuple(checkpoint_data["data_config"]["image_size"]),
        maximum_width=int(maximum_width),
    )
    affine_aligned, affine_flow = ink_bbox_affine_prealign(source_tensor, target_tensor)
    affine_displacement = torch.linalg.vector_norm(affine_flow, dim=1).mean().item()
    mean_displacement = torch.linalg.vector_norm(output.flow, dim=1).mean().item()
    confidence = output.confidence.mean().item()
    inference_mode = (
        f"tiled ({checkpoint_data['data_config']['image_size'][1]}px overlapping blocks)"
        if output.flow.shape[-1] > checkpoint_data["data_config"]["image_size"][1]
        else "single block"
    )
    details = (
        f"Inference mode: ink-affine prealignment + {inference_mode}\n"
        f"Mean affine displacement: {affine_displacement:.2f}px\n"
        f"Mean displacement: {mean_displacement:.2f}px\n"
        f"Mean patch confidence: {confidence:.3f}"
    )
    return (
        tensor_to_pil(affine_aligned),
        alignment_overlay(affine_aligned, target_tensor),
        tensor_to_pil(output.aligned),
        alignment_overlay(output.aligned, target_tensor),
        flow_visualization(output.flow),
        details,
    )


with gr.Blocks(title="Manuscript Spatial Registration") as app:
    gr.Markdown(
        """
        # Manuscript Spatial Registration
        The model predicts a dense target-to-source displacement field and uses a
        differentiable spatial transformer to produce the aligned source image.
        """
    )
    with gr.Row():
        source_input = gr.Image(type="pil", label="Source line (Is)")
        target_input = gr.Image(type="pil", label="Target line (It)")
    with gr.Row():
        checkpoint_input = gr.Textbox(value=str(DEFAULT_CHECKPOINT), label="Registration checkpoint")
        maximum_width = gr.Slider(512, 4096, value=2048, step=128, label="Maximum processing width")
    align_button = gr.Button("Align source to target", variant="primary")
    gr.Markdown("## Global affine stage")
    with gr.Row():
        affine_output = gr.Image(label="Affine-prealigned source")
        affine_overlay_output = gr.Image(label="Affine overlay: target vs source")
    gr.Markdown("## Final affine + dense registration")
    with gr.Row():
        aligned_output = gr.Image(label="Final aligned source (Ialigned)")
        overlay_output = gr.Image(label="Final overlay: target vs aligned source")
        flow_output = gr.Image(label="Predicted displacement field")
    details_output = gr.Textbox(label="Diagnostics")
    align_button.click(
        run_alignment,
        inputs=[source_input, target_input, checkpoint_input, maximum_width],
        outputs=[
            affine_output,
            affine_overlay_output,
            aligned_output,
            overlay_output,
            flow_output,
            details_output,
        ],
    )


if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())
