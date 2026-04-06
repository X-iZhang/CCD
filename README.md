<!-- Add logo here -->
<h1 align="center">
  <img src="./assets/CCD_icon_logo.png" alt="CCD Logo" height="27" style="position: relative; top: -2px;"/>
  <strong>CCD: Mitigating Hallucinations in Radiology MLLMs via Clinical Contrastive Decoding</strong>
</h1>


<div align="center">

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&center=true&width=500&lines=Training-free%2C+decode+smart.;Think+clinically%2C+act+precisely.;Powered+by+Clinical+Contrastive+Decoding."
       alt="Typing SVG"
       style="margin-bottom:-10px; display:block;" />
</a>

[![Project Page](https://img.shields.io/badge/Project-Page-4285F4?style=for-the-badge&logo=googlelens&logoColor=4285F4)](https://x-izhang.github.io/CCD/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.23379-b31b1b?style=for-the-badge&logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2509.23379)
[![Gradio Demo](https://img.shields.io/badge/Gradio-Online_Demo-FFCC00?style=for-the-badge&logo=gradio&logoColor=yellow&labelColor=grey)](https://huggingface.co/spaces/X-iZhang/CCD)
[![Evaluation Split](https://img.shields.io/badge/HF-Test_Dataset-AECBFA?style=for-the-badge&logo=huggingface&logoColor=FFCC00&labelColor=grey)](https://huggingface.co/collections/X-iZhang/ccd-68b9f5db2f03525b465ee09c)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://github.com/X-iZhang/CCD/blob/main/LICENSE)
[![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2FX-iZhang%2FCCD&label=Views&countColor=%23f36f43&style=for-the-badge)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FX-iZhang%2FCCD)

</div>

## 🔥 News
- **[06 Apr 2026]** 🎉 [***The paper***](https://arxiv.org/abs/2509.23379) has been accepted to [**ACL 2026**](https://2026.aclweb.org/)!
- **[02 Dec 2025]** 🧲 Added [**zero-shot device detection**](#supported-expert-models) (13 types) powered by [MedSigLIP](https://huggingface.co/google/medsiglip-448).
- **[17 Oct 2025]** 🔩 **CCD** has been upgraded to support **view classification** for chest X-rays — see the [Supported Expert Models](#supported-expert-models) section for details.
- **[06 Oct 2025]** 🎮 The **online demo** is available at [Hugging Face Spaces](https://huggingface.co/spaces/X-iZhang/CCD). Feel free to try it out!
- **[30 Sep 2025]** 🗂️ The processed test data for quick start are now available — enjoy exploring with [the provided guidelines](https://github.com/X-iZhang/CCD/tree/main?tab=readme-ov-file#%EF%B8%8F-dataset)!
- **[27 Sep 2025]** ⛳ Our preprint is now live on [arXiv](https://arxiv.org/abs/2509.23379) — check it out for details.

### 🎯 Call for Contribution
> We welcome contributions from the community to **`CDD+`** ! If you have ideas for new features or improvements, feel free to open an issue or contact us directly. We are especially interested in contributions that extend **`CCD`** to more label modalities, such as morphology (size/shape), anatomical location, and devices/lines/tubes.

<div align="center">
<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&center=true&color=F77D67&width=435&lines=CCD%2B+coming+soon!+%F0%9F%9A%80+"
       alt="Typing SVG"
       style="margin-bottom:-10px; display:block;" />
</a>
</div>

## Overview
Multimodal large language models (MLLMs) are advancing radiology by combining image and text understanding, but often generate inaccurate or unsupported clinical details—so-called medical hallucinations. We propose **C**linical **C**ontrastive **D**ecoding (**CCD**), a *training-free* and *retrieval-free* inference framework that integrates structured clinical signals from task‑specific radiology expert models. **`CCD`** reduces hallucinations and improves clinical accuracy without changing the base model. Experiments show **`CCD`** boosts performance on multiple datasets and models, offering a practical way to make radiology MLLMs more reliable.

<details open>
<summary>CCD's Framework</summary>

![framework](./assets/CCD_framework_new.png)

</details>

## 📖 Contents
- [⛏️ Installation](#️-installation)
- [⚡ Quick Start](#-quick-start)
  - [CLI Inference](#cli-inference)
  - [Script Inference](#script-inference)
  - [Gradio Web Interface](#gradio-web-interface)
- [🛠️ Advanced Usage](#️-advanced-usage)
  - [Supported MLLM Models](#supported-mllm-models)
  - [Supported Expert Models](#supported-expert-models)
  - [Parameter Settings](#parameter-settings)
- [🗂️ Dataset](#️-dataset)
- [📊 Evaluation](#-evaluation)
- [📝 Citation](#-citation)
- [📚 Acknowledgments](#-acknowledgments)
- [📨 Contact](#-contact)
- [📜 License](#-license)
- [🧰 Intended Use](#-intended-use)

## ⛏️ Installation

> [!TIP]  
> Use [`uv`](https://pypi.org/project/uv) for installation — it's faster and more reliable than `pip`.

### Option 1:
Install the latest version directly from GitHub for quick setup:

```bash
uv pip install git+https://github.com/X-iZhang/CCD.git
```

> [!NOTE]
> Requirements: Python 3.9 or later, and a CUDA-compatible GPU (recommended)

### Option 2: 
If you plan to modify the code or contribute to the project, you can clone the repository and install it in editable mode:

1. Clone the repository and navigate to the project folder

```bash
git clone https://github.com/X-iZhang/CCD.git
cd CCD
```

2. Set up the environment and install in editable mode

```bash
conda create -n CCD python=3.10 -y
conda activate CCD
pip install uv # enable uv support
uv pip install -e .
```

<details>
<summary> 🔄 Upgrade to the latest code base </summary>

```Shell
git pull
uv pip install -e .
```

</details>

## ⚡ Quick Start

### CLI Inference  
You can perform inference directly from the command line using our CLI tool:

```Shell
python -m ccd.run_ccd \
  --model-path "X-iZhang/libra-maira-2" \
  --image "./path/to/Chest_Xray.jpg" \
  --question "Is there evidence of any abnormalities?" \
  --max-new-tokens 128
```

**Optional arguments:**  

| Argument | Description | Default |
|-----------|--------------|----------|
| `--alpha` | Clinical guidance weight (range: 0.0–1.0) | 0.5 |
| `--beta` | Expert token weight (range: 0.0–1.0) | 0.5 |
| `--gamma` | Token bias magnitude (range: 2, 5, 10) | 10 |
| `--expert-model` | Choice of expert model: `"DenseNet"`, `"MedSiglip"`, `"View"`, or `"Device"` | DenseNet |

### Script Inference
You can run inference programmatically using the `ccd_eval` function from `ccd/run_ccd.py`.  
After installing this repository, you can easily launch a model (either your own trained model or ours) locally or in Google Colab.  


```python
from ccd import ccd_eval

# Run CCD inference on a chest X-ray
output = ccd_eval(
    model_path="X-iZhang/libra-maira-2",  # or your custom radiology MLLM
    image="./path/to/Chest_Xray.jpg",
    question="Describe the findings in this chest X-ray.",
    alpha=0.5,        # Clinical guidance weight
    beta=0.5,         # Expert token weight
    gamma=10,         # Token bias magnitude
    temperature=0.9,  # Sampling temperature
    top_p=0.9,        # Nucleus sampling probability
    top_k=50,         # Top-k sampling
    expert_model="DenseNet",    # or "MedSiglip" or "View" or "Device"
    max_new_tokens=256
)
print(output)
```

<details>
<summary>💡 You can also use <code>run_eval</code> to test the original model output (without CCD).</summary>

```python
from ccd import run_eval

# Run standard inference without CCD
output = run_eval(
    model_path="X-iZhang/libra-maira-2",
    image="./path/to/Chest_Xray.jpg",
    question="Describe the findings in this chest X-ray.",
    max_new_tokens=128,
    num_beams=1
)
print(output)
```

> 👉 <code>run_eval</code> also supported batch inference using a list of images and questions.

</details>

### Gradio Web Interface  

You can launch the Gradio demo locally with:  

```bash
python -m ccd.app
```

- Or try it directly on [🤗 **Hugging Face Spaces** 🤗](https://huggingface.co/spaces/X-iZhang/CCD).

Once the Gradio web interface is launched, you can open it using the URL printed on your screen. You will notice that both the default MAIRA-2 model and the expert models are ready for setup, with more models available in the list. Simply upload a chest X-ray image, enter your question, and click `🚀Generate` to view the results!

![demo](./assets/demo.gif)

## 🛠️ Advanced Usage

### Supported MLLM Models

CCD is compatible with any radiology MLLM that follows the **Libra/LLaVA** architecture:

> [!NOTE]  
> To switch MLLM models, simply set the `--model-path` argument (CLI) or `model_path` parameter (Python) to one of the following checkpoints.  

| Model | Checkpoint |
|--------|-------------|
| **Libra-v1.0-7B** | [X-iZhang/libra-v1.0-7b](https://huggingface.co/X-iZhang/libra-v1.0-7b) |
| **Libra-v1.0-3B** | [X-iZhang/libra-v1.0-3b](https://huggingface.co/X-iZhang/libra-v1.0-3b) |
| **MAIRA-2** | [X-iZhang/libra-maira-2](https://huggingface.co/X-iZhang/libra-maira-2) |
| **LLaVA-Med-v1.5** | [X-iZhang/libra-llava-med-v1.5-mistral-7b](https://huggingface.co/X-iZhang/libra-llava-med-v1.5-mistral-7b) |
| **LLaVA-Rad** | [X-iZhang/libra-llava-rad](https://huggingface.co/X-iZhang/libra-llava-rad) |
| **Med-CXRGen-F** | [X-iZhang/Med-CXRGen-F](https://huggingface.co/X-iZhang/Med-CXRGen-F) |
| **Med-CXRGen-I** | [X-iZhang/Med-CXRGen-I](https://huggingface.co/X-iZhang/Med-CXRGen-I) |

> [!WARNING]  
> The model adapted from the [**Libra**](https://github.com/X-iZhang/Libra) repository is intended for demonstration purposes only. For accurate evaluation, please refer to the original model weights and configuration settings, particularly the chat template.

### Supported Expert Models

CCD integrates four expert models' signals for **clinical signal extraction**:  

> [!NOTE]  
> To switch expert models, simply set the `--expert-model` argument (CLI) or `expert_model` parameter (Python) to one of the following names.  

| Model | Checkpoint | Note |
|--------|-------------|------|
| **DenseNet** | [torchxrayvision/densenet121-res224-chex](https://huggingface.co/torchxrayvision/densenet121-res224-chex) | CheXpert (Stanford)|
| **MedSiglip** | [google/medsiglip-448](https://huggingface.co/google/medsiglip-448) |  Variant of [SigLIP](https://arxiv.org/abs/2303.15343) |
| **View Model** | [ChestViewSplit](https://github.com/xinario/chestViewSplit) |  'Frontal' or 'Lateral' |
| **Device Model** | [google/medsiglip-448](https://huggingface.co/google/medsiglip-448)  |  Zero-shot detection of 13 device types or 'No Device'. |

> [!TIP]  
> When deploying *DenseNet*, it has been upgraded to support the *view classification expert model*, which helps the system better understand the view position of chest X-rays, thereby improving the accuracy of report generation. *MedSigLIP* has also been configured accordingly. The design is inspired by the [MAIRA-2 chat template](https://huggingface.co/microsoft/maira-2/blob/main/processing_maira2.py#L191).

### Parameter Settings

- **`alpha` (0.0-1.0)**: Weight for clinical guidance text
  - Higher = more influence from expert-generated guidance
  - Recommended: 0.3-0.7

- **`beta` (0.0-1.0)**: Weight for direct token biasing
  - Higher = stronger push toward clinical terminology
  - Recommended: 0.3-0.7

- **`gamma` (2, 5, 10)**: Maximum token bias magnitude
  - 2: Subtle influence
  - 5: Moderate influence
  - 10: Strong influence (default)

> [!TIP]  
> These parameters can be set beyond the recommended range for adversarial testing to observe CCD’s behaviour under extreme conditions.

## 🗂️ Dataset  

CCD supports multiple medical imaging datasets commonly used in radiology research:  

- [**MIMIC-CXR**](https://physionet.org/content/mimic-cxr/) — Chest X-ray images with corresponding radiology reports.  
- [**IU-Xray**](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) — Chest X-ray dataset with structured annotations.  
- [**CheXpert Plus**](https://aimi.stanford.edu/datasets/chexpert-plus) — Large-scale dataset for chest X-ray interpretation.  
- [**Medical-CXR-VQA**](https://physionet.org/content/medical-cxr-vqa-dataset/) — A dataset for visual question answering in chest X-rays.

> [!NOTE]  
> To facilitate hands-on testing, we provide pre-processed test splits for [**MIMIC-CXR**](https://huggingface.co/datasets/X-iZhang/MIMIC-CXR-RRG), [**IU-Xray**](https://huggingface.co/datasets/X-iZhang/IU-Xray-RRG), [**CheXpert Plus**](https://huggingface.co/datasets/X-iZhang/CheXpert-plus-RRG) and [**Medical-CXR-VQA**](https://huggingface.co/datasets/X-iZhang/Medical-CXR-VQA), available on [Hugging Face Collections](https://huggingface.co/collections/X-iZhang/ccd-68b9f5db2f03525b465ee09c).

> [!WARNING]
> Carefully read the `READMEs`; Please note that the image quality of these datasets has been compressed for efficient storage and sharing. Use the original datasets for evaluation.

## 📊 Evaluation  

For evaluating generated reports, we recommend using [**RadEval**](https://pypi.org/project/RadEval/) — a unified framework for radiology text evaluation that integrates multiple standard metrics. Details can be found in the [GitHub repository](https://github.com/jbdel/RadEval).

You can install RadEval via pip:  
```bash
pip install RadEval
```
> [!TIP]  
> RadEval supports metrics such as BLEU, ROUGE, BERTScore, CheXbert F1, and RadGraph F1, making it ideal for comprehensive evaluation of radiology report generation models.

## 📝 Citation

If you find our paper and code useful in your research and applications, please cite using this BibTeX:

```bibtex
@article{zhang2025ccd,
  title={CCD: Mitigating Hallucinations in Radiology MLLMs via Clinical Contrastive Decoding},
  author={Zhang, Xi and Meng, Zaiqiao and Lever, Jake and Ho, Edmond SL},
  journal={arXiv preprint arXiv:2509.23379},
  year={2025}
}
```

## 📚 Acknowledgments

This project builds upon the following outstanding open-source works:

- [**Libra**](https://github.com/X-iZhang/Libra) — A flexible toolkit supporting multiple radiology LLM backbones, covering the full pipeline from training to inference.
- [**TorchXRayVision**](https://github.com/mlmed/torchxrayvision) — A library for chest X-ray datasets and models. 
- [**MedSigLIP**](https://huggingface.co/google/medsiglip-448) — Medical Signal–Language Image Pretraining.
- [**RadEval**](https://pypi.org/project/RadEval/) — A unified framework for radiology text evaluation.

We thank the authors for their valuable contributions to the medical AI community.

## 📨 Contact
For any enquiries or collaboration opportunities, please contact: [**x.zhang.6@research.gla.ac.uk**](mailto:x.zhang.6@research.gla.ac.uk)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🧰 Intended Use

**CCD** is designed to **assist** clinical practitioners, researchers, and medical trainees in generating and analysing **chest X-ray reports**, with a focus on **temporal reasoning** and **context-aware description** of radiological findings.

### Key Applications

- 🩺 **Clinical Decision Support** — Produces preliminary *findings* or *comparative analyses* that can aid radiologists in drafting and reviewing reports.  
- 🎓 **Educational Tool** — Demonstrates example interpretations and temporal progressions for teaching radiology residents and students.  
- 🔬 **Research Utility** — Enables investigation of automated report generation, visual-language alignment, and temporal feature learning in medical imaging.

>[!IMPORTANT]
> All outputs must be reviewed and validated by **qualified radiologists or medical professionals** before informing any clinical decision.

---

<details>
<summary><strong>Limitations and Recommendations</strong></summary>

1. **Data Bias** — Performance may degrade on underrepresented populations or rare disease categories.  
2. **Clinical Oversight** — CCD is a *supportive* system, not a replacement for professional medical judgment.  
3. **Temporal Sensitivity** — Although TAC enhances temporal alignment, subtle or atypical longitudinal changes may remain unrecognised.  
4. **Generalisation** — Performance may vary on image types or clinical contexts not present in the training distribution.  

</details>

<details>
<summary><strong>Ethical Considerations</strong></summary>

- **Patient Privacy** — All input data must be fully de-identified and compliant with **HIPAA**, **GDPR**, or equivalent local regulations.  
- **Responsible Deployment** — CCD’s outputs may contain inaccuracies; users should interpret them with appropriate caution.  
- **Accountability** — The responsibility for clinical verification and safe deployment lies with the **end-user organisation or researcher**.  

</details>

<details>
<summary><strong>Disclaimer</strong></summary>

This model and accompanying tools are intended **solely for research and educational purposes**.  
CCD is **not approved** by the **FDA**, **CE**, or other regulatory authorities for clinical use.  
For medical diagnosis or treatment decisions, please consult a **licensed healthcare professional**.  

</details>