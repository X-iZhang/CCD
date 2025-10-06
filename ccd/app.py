import os
import torch
import gradio as gr
from ccd import ccd_eval, run_eval
from libra.eval.run_libra import load_model


# =========================================
# Global Configuration
# =========================================
MODEL_CATALOGUE = {
    "Libra-v1.0-7B": "X-iZhang/libra-v1.0-7b",
    "Libra-v1.0-3B": "X-iZhang/libra-v1.0-3b",
    "MAIRA-2": "X-iZhang/libra-maira-2",
    "LLaVA-Med-v1.5": "X-iZhang/libra-llava-med-v1.5-mistral-7b",
    "LLaVA-Rad": "X-iZhang/libra-llava-rad",
    "Med-CXRGen-F": "X-iZhang/Med-CXRGen-F",
    "Med-CXRGen-I": "X-iZhang/Med-CXRGen-I"
}
DEFAULT_MODEL_NAME = "MAIRA-2"
_loaded_models = {}


# =========================================
# Environment Setup
# =========================================
def setup_environment():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_CACHE'] = './cache'
    torch.set_num_threads(4)


# =========================================
# Model Loader
# =========================================
def load_or_get_model(model_name: str):
    """Load the model based on its display name."""
    model_path = MODEL_CATALOGUE[model_name]
    if model_path in _loaded_models:
        return _loaded_models[model_path]

    print(f"🔹 Loading model: {model_name} ({model_path}) ...")
    with torch.no_grad():
        model = load_model(model_path)
    _loaded_models[model_path] = model
    print(f"✅ Loaded successfully: {model_name}")
    return model


# =========================================
# CCD Logic
# =========================================
def generate_ccd_description(
    selected_model_name,
    current_img,
    prompt,
    expert_model,
    alpha,
    beta,
    gamma,
    use_run_eval,
    max_new_tokens
):
    """Generate findings using CCD evaluation."""
    if not current_img:
        return "⚠️ Please upload or select an example image first."

    try:
        model = load_or_get_model(selected_model_name)
        ccd_output = ccd_eval(
            libra_model=model,
            image=current_img,
            question=prompt,
            max_new_tokens=max_new_tokens,
            expert_model=expert_model,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )

        if use_run_eval:
            baseline_output = run_eval(
                libra_model=model,
                image=current_img,
                question=prompt,
                max_new_tokens=max_new_tokens,
                num_beams=1
            )
            return (
                f"### 🩺 CCD Result ({expert_model})\n{ccd_output}\n\n"
                f"---\n### ⚖️ Baseline (run_eval)\n{baseline_output}"
            )

        return f"### 🩺 CCD Result ({expert_model})\n{ccd_output}"

    except Exception:
        import traceback, sys
        error_msg = traceback.format_exc()
        print("========== CCD ERROR LOG ==========", file=sys.stderr)
        print(error_msg, file=sys.stderr)
        print("===================================", file=sys.stderr)
        return f"❌ Exception Trace:\n```\n{error_msg}\n```"


# =========================================
# Main Application
# =========================================
def main():
    setup_environment()

    # Example Image Path
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    example_path = os.path.abspath(os.path.join(cur_dir, "..", "assets", "example.jpg"))
    example_exists = os.path.exists(example_path)

    # Model reference table
    model_table = """
| **Model Name** | **HuggingFace Link** |
|----------------|----------------------|
| **Libra-v1.0-7B** | [X-iZhang/libra-v1.0-7b](https://huggingface.co/X-iZhang/libra-v1.0-7b) |
| **Libra-v1.0-3B** | [X-iZhang/libra-v1.0-3b](https://huggingface.co/X-iZhang/libra-v1.0-3b) |
| **MAIRA-2** | [X-iZhang/libra-maira-2](https://huggingface.co/X-iZhang/libra-maira-2) |
| **LLaVA-Med-v1.5** | [X-iZhang/libra-llava-med-v1.5-mistral-7b](https://huggingface.co/X-iZhang/libra-llava-med-v1.5-mistral-7b) |
| **LLaVA-Rad** | [X-iZhang/libra-llava-rad](https://huggingface.co/X-iZhang/libra-llava-rad) |
| **Med-CXRGen-F** | [X-iZhang/Med-CXRGen-F](https://huggingface.co/X-iZhang/Med-CXRGen-F) |
| **Med-CXRGen-I** | [X-iZhang/Med-CXRGen-I](https://huggingface.co/X-iZhang/Med-CXRGen-I) |
    """

    with gr.Blocks(title="📷 Clinical Contrastive Decoding", theme="soft") as demo:
        gr.Markdown("""
        # 📷 CCD: Mitigating Hallucinations in Radiology MLLMs via Clinical Contrastive Decoding
        Generate radiology findings using **CCD Evaluation** with multiple base models.
        Upload or use an example chest X-ray image, choose a model, and configure CCD parameters.
        """)

        with gr.Tab("✨ CCD Demo"):
            with gr.Row():
                # -------- Left Column: Image --------
                with gr.Column(scale=1):
                    gr.Markdown("### Radiology Image (eg. Chest X-ray)")
                    current_img = gr.Image(label="Radiology Image", type="filepath", interactive=True)
                    if example_exists:
                        gr.Examples(
                            examples=[[example_path]],
                            inputs=[current_img],
                            label="Example Image"
                        )
                    else:
                        gr.Markdown(f"⚠️ Example image not found at `{example_path}`")

                # -------- Right Column: Controls --------
                with gr.Column(scale=1):
                    gr.Markdown("### Model Selection & Prompt")
                    selected_model_name = gr.Dropdown(
                        label="Base Radiology MLLM",
                        choices=list(MODEL_CATALOGUE.keys()),
                        value=DEFAULT_MODEL_NAME
                    )
                    prompt = gr.Textbox(
                        label="Question / Prompt",
                        value="What are the findings in this chest X-ray?",
                        lines=3
                    )

                    gr.Markdown("### CCD Parameters")
                    expert_model = gr.Radio(
                        label="Expert Model",
                        choices=["MedSigLip", "DenseNet"],
                        value="DenseNet"
                    )
                    with gr.Row():
                        alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Alpha")
                        beta = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Beta")
                        gamma = gr.Slider(0, 20, value=10, step=1, label="Gamma")

                    with gr.Accordion("Advanced Options", open=False):
                        max_new_tokens = gr.Slider(10, 500, value=128, step=1, label="Max New Tokens")
                        use_run_eval = gr.Checkbox(label="Compare with baseline (run_eval)", value=False)

                    generate_btn = gr.Button("🚀 Generate", variant="primary")

            # -------- Output --------
            output = gr.Markdown(label="Output", value="Results will appear here.")
            generate_btn.click(
                fn=generate_ccd_description,
                inputs=[
                    selected_model_name, current_img, prompt,
                    expert_model, alpha, beta, gamma,
                    use_run_eval, max_new_tokens
                ],
                outputs=output
            )

        # -------- Model Table --------
        # gr.Markdown("### 🧠 Supported Models")
        # gr.Markdown(model_table)

        gr.Markdown("""
        ---
        **Note:** CCD integrates expert model feedback (e.g., DenseNet or MedSigLip) 
        to refine radiology report generation.
        """)

    demo.launch(debug=True)


if __name__ == "__main__":
    main()