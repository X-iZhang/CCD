<!-- Add logo here -->
<h1 align="center">
  <img src="./assets/CCD_icon_logo.png" alt="CCD Logo" height="27" style="position: relative; top: -2px;"/>
  <strong>CCD: Mitigating Hallucinations in Radiology MLLMs via Clinical Contrastive Decoding</strong>
</h1>

<h3 align="center">
  <strong>🤖 We're still cooking — Stay tuned! 🤖<br>⭐ Give us a star if you like it! ⭐</strong>
</h3>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-Green?logo=stylelint)](https://x-izhang.github.io/CCD/)
[![Gradio Demo](https://img.shields.io/badge/-Online%20Demo-yellow.svg?logo=gradio&labelColor=grey)](https://huggingface.co/spaces/X-iZhang/CCD)
[![Evaluation Split](https://img.shields.io/badge/-Test_Dataset-AECBFA?logo=googlecloudstorage&labelColor=grey)](https://huggingface.co/collections/X-iZhang/ccd-68b9f5db2f03525b465ee09c)
[![arXiv](https://img.shields.io/badge/Arxiv-preprint-b31b1b.svg?logo=arXiv)]() 
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?)](https://github.com/X-iZhang/CCD/blob/main/LICENSE)
[![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2FX-iZhang%2FCCD&label=Views&countColor=%23f36f43&style=flat)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FX-iZhang%2FCCD)

</div>

## 🔥 News
- **[27 Sep 2025]** ⛳ Our preprint is now live on [arXiv]() — check it out for details.

## Overview
Multimodal large language models (MLLMs) are advancing radiology by combining image and text understanding, but often generate inaccurate or unsupported clinical details—so-called medical hallucinations. We propose **C**linical **C**ontrastive **D**ecoding (**CCD**), a *training-free* and *retrieval-free* inference framework that integrates structured clinical signals from task‑specific radiology expert models. CCD reduces hallucinations and improves clinical accuracy without changing the base model. Experiments show CCD boosts performance on multiple datasets and models, offering a practical way to make radiology MLLMs more reliable.
<details open>
<summary>CCD’s Framework</summary>

![framework](./assets/CCD_framework_new.png)

</details>

## Citation ✒️

If you find our paper and code useful in your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{zhang2026ccd,
  title={CCD: Mitigating Hallucinations in Radiology MLLMs via Clinical Contrastive Decoding},
  author={Zhang, Xi and Meng, Zaiqiao and Lever, Jake and Ho, Edmond S. L.},
  booktitle=tbd,
  year={2025},
  url={https://x-izhang.github.io/CCD/}
}
```