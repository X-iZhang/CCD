import os
import torch
import gradio as gr
import time
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
    if torch.cuda.is_available():
        print("🔹 Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("🔹 Using CPU")
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_CACHE'] = './cache'
    torch.set_num_threads(4)


# =========================================
# Model Loader
# =========================================
def load_or_get_model(model_name: str):
    """Load the model based on its display name."""
    model_path = MODEL_CATALOGUE[model_name]
    print(f"🔹 Model path resolved: {model_path}")
    if model_path in _loaded_models:
        print(f"🔹 Model already loaded: {model_name}")
        return _loaded_models[model_path]

    print(f"🔹 Loading model: {model_name} ({model_path}) ...")
    try:
        with torch.no_grad():
            model = load_model(model_path)
        _loaded_models[model_path] = model
        print(f"✅ Loaded successfully: {model_name}")
        return model
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        raise


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
        print(f"🔹 Generating description with model: {selected_model_name}")
        print(f"🔹 Parameters: alpha={alpha}, beta={beta}, gamma={gamma}")
        print(f"🔹 Image path: {current_img}")

        model = load_or_get_model(selected_model_name)
        print(f"🔹 Running CCD with {selected_model_name} and expert model {expert_model}...")
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
                f"---\n### ⚖️ Baseline (run_eval)\n{baseline_output[0]}"
            )

        return f"### 🩺 CCD Result ({expert_model})\n{ccd_output}"

    except Exception:
        import traceback, sys
        error_msg = traceback.format_exc()
        print("========== CCD ERROR LOG ==========", file=sys.stderr)
        print(error_msg, file=sys.stderr)
        print("===================================", file=sys.stderr)
        return f"❌ Exception Trace:\n```\n{error_msg}\n```"


def safe_generate_ccd_description(
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
    """Wrapper around generate_ccd_description that logs inputs and prints full traceback on error."""
    import traceback, sys, time
    print("\n=== Gradio callback invoked ===")
    print(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"selected_model_name={selected_model_name}")
    print(f"current_img={current_img}")
    print(f"prompt={prompt}")
    print(f"expert_model={expert_model}, alpha={alpha}, beta={beta}, gamma={gamma}, use_run_eval={use_run_eval}, max_new_tokens={max_new_tokens}")

    try:
        return generate_ccd_description(
            selected_model_name,
            current_img,
            prompt,
            expert_model,
            alpha,
            beta,
            gamma,
            use_run_eval,
            max_new_tokens
        )
    except Exception as e:
        err = traceback.format_exc()
        print("========== GRADIO CALLBACK ERROR ==========", file=sys.stderr)
        print(err, file=sys.stderr)
        print("==========================================", file=sys.stderr)
        # Also write the error and inputs to a persistent log file for easier inspection
        try:
            with open('/workspace/CCD/callback.log', 'a', encoding='utf-8') as f:
                f.write('\n=== CALLBACK LOG ENTRY ===\n')
                f.write(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"selected_model_name={selected_model_name}\n")
                f.write(f"current_img={current_img}\n")
                f.write(f"prompt={prompt}\n")
                f.write(f"expert_model={expert_model}, alpha={alpha}, beta={beta}, gamma={gamma}, use_run_eval={use_run_eval}, max_new_tokens={max_new_tokens}\n")
                f.write('TRACEBACK:\n')
                f.write(err + '\n')
                f.write('=== END ENTRY ===\n')
        except Exception as fe:
            print(f"Failed to write callback.log: {fe}", file=sys.stderr)
        # Also return a user-friendly error message to the UI with traceback
        return f"❌ An internal error occurred. See server logs for details.\n\nTraceback:\n```\n{err}\n```"


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
        ### [Project Page](https://x-izhang.github.io/CCD/) | [Paper](https://arxiv.org/abs/2509.23379) | [Code](https://github.com/X-iZhang/CCD) | [Models](https://huggingface.co/collections/X-iZhang/libra-6772bfccc6079298a0fa5f8d)
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
                        value="What are the findings in this chest X-ray? Give a detailed description.",
                        lines=1
                    )

                    gr.Markdown("### CCD Parameters")
                    expert_model = gr.Radio(
                        label="Expert Model",
                        choices=["MedSigLip", "DenseNet"],
                        value="DenseNet"
                    )

                    # Notice for MedSigLip access requirements (hidden by default)
                    medsiglip_message = (
                        "**Note: The MedSigLip model requires authorization to access.**\n\n"
                        "To use MedSigLip, please deploy the Gradio Web Interface locally and complete the authentication steps.\n"
                        "See deployment instructions and how to run locally here: "
                        "[Gradio Web Interface](https://github.com/X-iZhang/CCD#gradio-web-interface)"
                    )
                    medsiglip_notice = gr.Markdown(value="", visible=False)

                    def _toggle_medsiglip_notice(choice):
                        if choice == "MedSigLip":
                            return gr.update(visible=True, value=medsiglip_message)
                        else:
                            return gr.update(visible=False, value="")

                    # Connect radio change to the notice visibility
                    expert_model.change(fn=_toggle_medsiglip_notice, inputs=[expert_model], outputs=[medsiglip_notice])

                    with gr.Row():
                        alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Alpha")
                        beta = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Beta")
                        gamma = gr.Slider(0, 20, value=10, step=1, label="Gamma")

                    with gr.Accordion("Advanced Options", open=False):
                        max_new_tokens = gr.Slider(10, 256, value=128, step=1, label="Max New Tokens")
                        use_run_eval = gr.Checkbox(label="Compare with baseline (run_eval)", value=False)

                    generate_btn = gr.Button("🚀 Generate", variant="primary")

            # -------- Output --------
            # output = gr.Markdown(label="Output", value="### 📷 Results will appear here.👇")
            output = gr.Markdown(
                value='<h3 style="color:#007BFF;">📷 Results will appear here.👇</h3>',
                label="Output"
            )
            # Switch callback to the safe wrapper
            generate_btn.click(
                fn=safe_generate_ccd_description,
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
        ### Terms of Use
        The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA.
        
        By accessing or using this demo, you acknowledge and agree to the following:
        - **Research & Non-Commercial Purposes**: This demo is provided solely for research and demonstration. It must not be used for commercial activities or profit-driven endeavors.
        - **Not Medical Advice**: All generated content is experimental and must not replace professional medical judgment.
        - **Content Moderation**: While we apply basic safety checks, the system may still produce inaccurate or offensive outputs.
        - **Responsible Use**: Do not use this demo for any illegal, harmful, hateful, violent, or sexual purposes.
        By continuing to use this service, you confirm your acceptance of these terms. If you do not agree, please discontinue use immediately.
        """)


    # Log that Gradio is starting (helpful when stdout/stderr are captured)
    try:
        with open('/workspace/CCD/callback.log', 'a', encoding='utf-8') as f:
            f.write(f"\n=== GRADIO START ===\nstarted_at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    except Exception:
        pass

    # Bind to 0.0.0.0 so the server is reachable from host/container and set an explicit port
    demo.launch(share=True)


if __name__ == "__main__":
    main()