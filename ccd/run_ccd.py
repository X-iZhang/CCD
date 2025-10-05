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

import sys, time
import torch
import argparse
from libra.utils import disable_torch_init
from libra.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from libra.conversation import conv_templates, SeparatorStyle
from libra.eval.run_libra import load_images, load_model, get_image_tensors_batch
from libra.mm_utils import tokenizer_image_token_batch, get_model_name_from_path
from libra.eval import libra_eval_batch 

from .ccd_utils import (
    predict_medsiglip_labels,
    predict_chexpert_labels,
    clinical_guide_generation,
    ccd
)

from functools import lru_cache
import transformers
transformers.logging.set_verbosity_error()

special_tokens = [
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
]

@lru_cache(maxsize=2)
def get_libra_model(model_path, model_base=None):
    model = load_model(model_path, model_base)
    return model

_model_cache = {} 

def run_eval(
    model_path=None,
    model_base=None,
    image=None,                   
    question=None,                  
    conv_mode=None,
    temperature=0.0,
    top_p=None,
    top_k=None,
    num_beams=1,
    length_penalty=1.0,
    max_new_tokens=128,
    libra_model=None,
    num_return_sequences=1,
    no_repeat_ngram_size=3
):
    """Wrapper for CCD: internally calls libra's libra_eval_batch function"""
    if isinstance(image, str):
        image = load_images(image)

    images = [image]
    queries = [question]
    assert isinstance(images, list) and isinstance(queries, list), "images and queries must be lists"
    assert len(images) == len(queries), "images and queries must be of the same length"

    disable_torch_init()

    # ✅ Automatically use cached model (libra_model argument takes precedence if provided)
    model = libra_model or get_libra_model(model_path, model_base)

    outputs = libra_eval_batch(
        model_path=model_path,
        model_base=model_base,
        images=images,                   # List of PIL images
        queries=queries,                  # List of query strings
        conv_mode=conv_mode,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        length_penalty=length_penalty,
        max_new_tokens=max_new_tokens,
        libra_model=model
    )

    return outputs
    

def ccd_eval(
    model_path=None,
    model_base=None,
    image=None,                   
    question=None,                  
    conv_mode=None,
    temperature=0.0,
    top_p=1.0,
    top_k=None,
    num_beams=1,
    length_penalty=1.0,
    max_new_tokens=128,
    libra_model=None,
    no_repeat_ngram_size=3,
    alpha=0.5,
    beta=0.5,
    gamma=10,
    expert_model="DenseNet"
):

    if isinstance(image, str):
        image = load_images(image)

    images = [image]
    queries = [question]
    assert isinstance(images, list) and isinstance(queries, list), "images and queries must be lists"
    assert len(images) == len(queries), "images and queries must be of the same length"

    disable_torch_init()

    # Model loading and caching logic
    global _model_cache
    if libra_model is not None:
        tokenizer, model, image_processor, context_len = libra_model
        model_name = model.config._name_or_path
    else:
        cache_key = model_path or "default"
        if cache_key in _model_cache:
            tokenizer, model, image_processor, context_len = _model_cache[cache_key]
            model_name = model.config._name_or_path
        else:
            tokenizer, model, image_processor, context_len = load_model(model_path, model_base)
            model_name = get_model_name_from_path(model_path)
            _model_cache[cache_key] = (tokenizer, model, image_processor, context_len)

    tokens_to_add = [t for t in special_tokens if t not in tokenizer.get_vocab()]

    if len(tokens_to_add) > 0:
        tokenizer.add_tokens(tokens_to_add)
        model.resize_token_embeddings(len(tokenizer))

    # Determine conversation mode based on model name
    if 'libra-v1.0-3b' in model_name.lower():
        mode_conv = "libra_llama_3"
    elif 'cxrgen' in model_name.lower():
        mode_conv = "libra_v0"
    elif 'mistral' in model_name.lower():
        mode_conv = "mistral_instruct"
    elif 'llava_med' in model_name.lower():
        mode_conv = "llava_med_v1.5_mistral_7b"
    elif 'maira' in model_name.lower():
        mode_conv = "maira_2"
    else:
        mode_conv = "libra_v1"

    if conv_mode is not None and mode_conv != conv_mode:
        print(f'[WARNING] Auto-inferred conversation mode is {mode_conv}, but `--conv-mode` is {conv_mode}, using {conv_mode}')
    else:
        conv_mode = mode_conv

    # Create prompts
    stop_strs = []
    prompts = []
    clinical_guides = []
    for q, img in zip(queries, images):
        # ===== 1. Build the original prompt =====
        if model.config.mm_use_im_start_end:
            q_orig = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + q
        else:
            q_orig = DEFAULT_IMAGE_TOKEN + '\n' + q

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], q_orig)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompts.append(prompt)

        # ===== 2. Predict CheXpert labels and build clinical guidance =====
        if expert_model.lower() == "medsiglip":
            label_score_dict = predict_medsiglip_labels(img)
        elif expert_model.lower() == "densenet":
            label_score_dict = predict_chexpert_labels(img)
        else:
            raise ValueError(f"❌ Unknown expert model: {expert_model}")

        clinical_guide = clinical_guide_generation(label_score_dict)
        
        # ===== 3. Build perturbed prompt (clinical_guides) =====
        if clinical_guide is not None:
            q_guidance = q_orig + " " + clinical_guide
        else:
            q_guidance = q_orig  # use original query if no guidance

        conv_dist = conv_templates[conv_mode].copy()
        conv_dist.append_message(conv_dist.roles[0], q_guidance)
        conv_dist.append_message(conv_dist.roles[1], None)
        guid_prompt = conv_dist.get_prompt()
        clinical_guides.append(guid_prompt)


    input_ids = tokenizer_image_token_batch(prompts, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()
    guidance_ids = tokenizer_image_token_batch(clinical_guides, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').cuda()

    image_tensors = get_image_tensors_batch(images, image_processor, model)

    stop_str = conv.sep if conv.sep_style not in {SeparatorStyle.TWO, SeparatorStyle.LLAMA_3, SeparatorStyle.MISTRAL} else conv.sep2
    stop_strs.append(stop_str)

    with torch.inference_mode():
        torch.cuda.empty_cache()
        if num_beams >= 2:
            sys.exit(
                "❌ Beam search is not supported for Clinical Contrastive Decoding.\n"
                "   Reason: computational overhead is too high in both stages.\n"
                "   👉 Please switch to greedy or sampling decoding instead."
            )
        else:
            _, ccd_output = ccd(
                label_score_dict=label_score_dict,
                model=model,
                input_ids=input_ids,
                guidance_ids=guidance_ids,
                image_tensors=image_tensors,
                tokenizer=tokenizer,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                keywords=stop_strs)

    return ccd_output 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Clinical Contrastive Decoding")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--question", type=str, required=True, help="Question to ask")
    parser.add_argument("--conv-mode", type=str, default=None, help="Conversation mode")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="Length penalty for generation")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3, help="No repeat n-gram size")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for CCD, [0,1]")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta for CCD, [0,1]")
    parser.add_argument("--gamma", type=float, default=10, help="Gamma for CCD, {2,5,10}")
    args = parser.parse_args()

    answer = ccd_eval(**vars(args))
    print("\n[📷 CCD Output]\n", end='', flush=True)

    for word in answer.split(" "):
        print(word + " ", end='', flush=True)
        time.sleep(0.12)
    print("\n")