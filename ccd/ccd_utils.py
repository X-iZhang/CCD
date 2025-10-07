#    Copyright (c) 2025 Xi Zhang
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

import torch
import math
import numpy as np
import torchvision
from typing import Dict
import torchxrayvision as xrv
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    AutoModel,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    NoRepeatNGramLogitsProcessor
)

_CHEXPERT = None
_MED_SIGILIP = None
_VIEW = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _get_chexpert_models():
    global _CHEXPERT, _VIEW
    if _CHEXPERT is None:
        _CHEXPERT = xrv.models.DenseNet(weights="densenet121-res224-chex").to(_DEVICE).eval()
    if _VIEW is None:
        _VIEW = xrv.baseline_models.xinario.ViewModel().to(_DEVICE).eval()
    return _CHEXPERT, _VIEW

def _get_med_sigilip_model():
    global _MED_SIGILIP
    if _MED_SIGILIP is None:
        model_id = "google/medsiglip-448"
        model = AutoModel.from_pretrained(model_id).to(_DEVICE)
        model.eval()  # Remember to call eval() during inference
        processor = AutoProcessor.from_pretrained(model_id)
        _MED_SIGILIP = (model, processor)
    return _MED_SIGILIP

# ====== Define labels and prompt construction ======
findings = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "Pleural Other",
    "Support Devices",
]

def make_prompts(label):
    return [
        f"a chest X-ray with {label.lower()}",
        f"a chest X-ray with no {label.lower()}"
    ]

def predict_medsiglip_labels(image):

    if image.mode != "RGB":
        image = image.convert("RGB")

    all_prompts = []
    label2indices = {}

    # Disease prompts
    for l in findings:
        pos, neg = make_prompts(l)
        label2indices[l] = (len(all_prompts), len(all_prompts)+1)
        all_prompts.extend([pos, neg])

    # View prompts
    view_prompts = [
        "Frontal view",
        "Lateral view"
    ]
    label2indices["Frontal"] = (len(all_prompts), len(all_prompts)+1)
    all_prompts.extend(view_prompts)

    MedSiglip_model, MedSiglip_processor = _get_med_sigilip_model()
    # Forward inference
    inputs = MedSiglip_processor(text=all_prompts, images=[image], padding="max_length", return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        outputs = MedSiglip_model(**inputs)
    logits = outputs.logits_per_image.squeeze(0)  # [num_prompts]

    # Compute probabilities for each label
    label_score_dict = {}
    for label, (pos_idx, neg_idx) in label2indices.items():
        pair_logits = torch.stack([logits[pos_idx], logits[neg_idx]])
        probs = torch.softmax(pair_logits, dim=0)
        label_score_dict[label] = probs[0].item() 

    # Split Frontal/Lateral into two separate keys
    label_score_dict["Frontal"] = label_score_dict.pop("Frontal")
    label_score_dict["Lateral"] = 1 - label_score_dict["Frontal"]

    return label_score_dict

def predict_chexpert_labels(image) -> dict:

    if image.mode != "RGB":
        image = image.convert("RGB")

    img = np.array(image)
    img = xrv.datasets.normalize(img, 255)  # Normalize to range expected by xrv

    # Add channel dimension
    if img.ndim == 2:
        img = img[None, ...]
    elif img.ndim == 3 and img.shape[2] == 3:
        img = img.mean(2)[None, ...]

    # Center crop and resize
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    img = transform(img)

    # Convert to tensor format
    img_tensor = torch.from_numpy(img).float()  # [1,H,W]
    img_tensor = img_tensor.unsqueeze(0).to(_DEVICE)  # [B=1, 1, H, W]

    # Predict using the CheXpert model
    chex, view_model = _get_chexpert_models()
    with torch.inference_mode():
        logits_path = chex(img_tensor)                 # [1, C]
        scores = logits_path[0].detach().cpu().numpy()
        image_labels = chex.pathologies

        view_logits = view_model(img_tensor)           # [1, 2]
        view_probs = F.softmax(view_logits, dim=1)[0].detach().cpu().numpy()


    label_score_dict = {label: float(score) for label, score in zip(image_labels, scores) if label}
    for label, prob in zip(view_model.targets, view_probs):
        label_score_dict[label] = float(prob)

    return label_score_dict

def get_expert_model_token_score_dict(
    label_score_dict: Dict[str, float],
    tokenizer,
    threshold: float = 0.0,
    only_first_token: bool = True,
    add_leading_space: bool = False,
    decay_gamma: float = 0.95, 
) -> Dict[int, float]:
    """
    Obtain token IDs and their influence scores from CheXpert label scores.

    Args:
    - label_score_dict: dict[str, float], CheXpert labels and their scores
    - tokenizer: HuggingFace tokenizer
    - threshold: float, threshold for considering positive vs negative scores
    - only_first_token: bool, whether to use only the first token of each word (default True)

    Returns:
    - Dict[int, float]: mapping token_id -> score (positive for boosting, negative for suppression)
    """
    token_score_dict: Dict[int, float] = {}

    for label, score in label_score_dict.items():

        base_score = score if score >= threshold else -score

        for variant_label in {label, label.lower()}:
            text_label = variant_label.replace("_", " ")

            if add_leading_space:
                text_label = " " + text_label

            tokens = tokenizer.tokenize(text_label)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            if not token_ids:
                continue

            prev_was_space = True
            score_i = base_score
            for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
                include = (tok.startswith("Ġ") or tok.startswith("▁") or prev_was_space) if only_first_token else True

                if include:
                    if tid not in token_score_dict or abs(score_i) > abs(token_score_dict[tid]):
                        token_score_dict[tid] = float(score_i)

                score_i *= decay_gamma
                prev_was_space = tok in {"Ġ", "▁"}

    return token_score_dict

def clinical_guide_generation(label_score_dict, threshold=0.5):
    """
    Generate an English clinical guidance text from label scores.
    Frontal/Lateral labels are treated separately as view descriptions and are not mixed
    into clinical findings.
    """
    view_labels = []
    clinical_labels = []

    for label, score in label_score_dict.items():
        if score >= threshold:
            if label in ["Frontal", "Lateral"]:
                view_labels.append(label)
            else:
                clinical_labels.append(label)

    if not clinical_labels and not view_labels:
        return "No clinical guidance based on current findings."

    view_text = f"View type: {', '.join(view_labels)}." if view_labels else ""
    clinical_text = ""
    if clinical_labels:
        clinical_text = "Attention to the following clinical instructions: " + "; ".join(clinical_labels) + "."

    return clinical_text + " " + view_text

def ccd(label_score_dict,  model, tokenizer, image_tensors, 
        input_ids, guidance_ids,
        keywords, temperature, top_k, top_p, length_penalty, no_repeat_ngram_size,
        alpha=0.5, beta=0.5, gamma=10, 
        max_new_tokens=256,
        stop_on_keywords=True,
        do_sample=False,
        mode="logit"):

    token_score_dict = get_expert_model_token_score_dict(
        label_score_dict=label_score_dict,
        tokenizer=tokenizer,
    )
    
    generated_ids = input_ids.clone()
    dist_generated_ids = guidance_ids.clone()

    logits_processor = LogitsProcessorList([
        RepetitionPenaltyLogitsProcessor(penalty=length_penalty),
        NoRepeatNGramLogitsProcessor(no_repeat_ngram_size)
    ])
    logits_warper = LogitsProcessorList()
    
    if temperature and temperature != 1.0:
        logits_warper.append(TemperatureLogitsWarper(temperature))
    if top_k and top_k > 0:
        logits_warper.append(TopKLogitsWarper(top_k))
    if top_p and top_p < 1.0:
        logits_warper.append(TopPLogitsWarper(top_p))

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            logits_orig = model(input_ids=generated_ids, images=image_tensors, attention_mask=torch.ones_like(generated_ids)).logits[:, -1, :]
            logits_clinical = model(input_ids=dist_generated_ids, images=image_tensors, attention_mask=torch.ones_like(dist_generated_ids)).logits[:, -1, :]

        # ====== Convert to log-probabilities to remove scale/shift differences ======
        logits_orig = F.log_softmax(logits_orig, dim=-1)
        logits_clinical = F.log_softmax(logits_clinical, dim=-1)
        
        logits_soft_guide = (1 - alpha) * logits_orig + alpha * logits_clinical

        logits_expert = logits_soft_guide.clone()

        max_bias = math.log(gamma)
        for tid, score in token_score_dict.items():
            if mode == "logit":
                p = max(min(float(score), 1 - 1e-6), 1e-6)
                bias =  math.log(p / (1.0 - p))
            elif mode == "linear":
                bias = (2.0 * score - 1.0) * max_bias
            else:
                raise ValueError("mode must be 'logit' or 'linear'")
            if gamma > 1:
                bias = max(-max_bias, min(max_bias, bias))
            logits_expert[:, tid] = logits_expert[:, tid] + bias

        # Make generated_ids safe — keep negative sentinel values semantic but avoid CUDA errors
        safe_generated_ids = generated_ids.clone()
        safe_generated_ids[safe_generated_ids < 0] = 0  # Temporarily replace negative indices with 0 for logits ops

        logits_soft_guide = logits_processor(safe_generated_ids, logits_soft_guide)
        logits_soft_guide = logits_warper(safe_generated_ids, logits_soft_guide)
        
        logits_final = (1 - beta) * logits_soft_guide + beta * logits_expert

        if do_sample:
            probs = F.softmax(logits_final, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(logits_final, dim=-1, keepdim=True)

        next_token_text = tokenizer.decode(next_token_id[0])

        if stop_on_keywords and any(k in next_token_text for k in keywords):
            break

        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
        dist_generated_ids = torch.cat([dist_generated_ids, next_token_id], dim=1)

    ccd_output = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    ccd_output = ccd_output.strip()
    
    return "", ccd_output