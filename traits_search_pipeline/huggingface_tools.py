import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional
from functools import lru_cache

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
)

from huggingface_hub import login

login(token=open("secrets/huggingface.token", "r").read().strip())

SYSTEM_PROMPT = """
Output ONE JSON with keys exactly: "answer_text","reason","evidence".
"reason" in ["supported","not_enough_information","blank_page","content_unrelated","conflicting_evidence","non_english","page_error","model_uncertain"].
If unsure, output {"answer_text":"unknown","reason":"not_enough_information","evidence":[]}.

Rules for "answer_text":
- Infer the allowed value domain from the QUESTION.
- If the QUESTION clearly implies a fixed set (yes/no, life stages, specialist/generalist, etc.), restrict to that set + "unknown".
- If the QUESTION is categorical but no fixed set is obvious, output a single concise lowercase category string taken verbatim or nearly-verbatim from PAGE_TEXT. If none is supported, use "unknown".
- Never invent categories not grounded in PAGE_TEXT.

ENTITY CHECK:
- If PAGE_TEXT does NOT contain the focal entity (case-insensitive exact match or well-known synonym) → output exactly:
  {"answer_text":"unknown","reason":"content_unrelated","evidence":[]}
- Do NOT infer from related or different entities.

EVIDENCE RULES:
- When reason="supported", each evidence item must be a verbatim snippet from PAGE_TEXT that contains BOTH:
  (a) the focal entity (e.g., "harpalus sinensis" or its synonym), and
  (b) the claimed value/category.
- If no such snippet exists → do not claim "supported".

ANSWER SANITY:
- Do not repeat the entire list of options from the QUESTION.
- Output a single value from the allowed set or "unknown".

If the focal entity is absent in PAGE_TEXT → "content_unrelated".
If present but no categorical support → "not_enough_information".

General rules:
- Default to "unknown" unless PAGE_TEXT directly supports a value → then "reason":"supported".
- Blank/garbled → "blank_page" or "page_error".

QUESTION: {question}
PAGE_TEXT: {page_text}
Return JSON only.
""".strip()


@dataclass
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    do_sample: bool = True
    eos_patterns: Optional[list] = None  # optional extra stops


def supports_chat_template(tokenizer) -> bool:
    # Only treat as chat if a non-empty chat_template is present
    tmpl = getattr(tokenizer, "chat_template", None)
    return bool(tmpl)


def build_prompt(tokenizer, question: str, page_text: str) -> str:
    if supports_chat_template(tokenizer):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"QUESTION:\n{question}\n\nPAGE_TEXT:\n{page_text}\n\nReturn JSON only.",
            },
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # fallback for non-chat models like flan-t5
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"PAGE_TEXT:\n{page_text}\n\n"
        f"Return JSON only."
    )


def extract_first_json(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return None


def validate_schema(d: Dict[str, Any]) -> bool:
    if set(d.keys()) != {"answer_text", "reason", "evidence"}:
        return False
    if not isinstance(d["answer_text"], str):
        return False
    if not isinstance(d["reason"], str):
        return False
    if not isinstance(d["evidence"], list):
        return False
    return True


@lru_cache(maxsize=4)
def _load_model_and_tokenizer(model_name: str, device: str, dtype_tag: str):
    torch_dtype = torch.float16 if dtype_tag == "fp16" else torch.float32
    config = AutoConfig.from_pretrained(model_name)
    is_enc_dec = getattr(config, "is_encoder_decoder", False)
    if is_enc_dec:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map=device if device else None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map=device if device else None,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model.eval()
    return model, tokenizer


def _cached_model_tokenizer(model_name: str, device: Optional[str]):
    dtype_tag = "fp16" if torch.cuda.is_available() else "fp32"
    dev = device if device is not None else "auto"
    return _load_model_and_tokenizer(model_name, dev, dtype_tag)


def generate(
    model_name: str,
    question: str,
    page_text: str,
    device: Optional[str],
    cfg,
) -> Dict[str, Any]:
    model, tokenizer = _cached_model_tokenizer(model_name, device)

    def supports_chat_template(tokenizer) -> bool:
        tmpl = getattr(tokenizer, "chat_template", None)
        return bool(tmpl)

    def build_prompt(tokenizer, question: str, page_text: str) -> str:
        system_prompt = (
            'Output ONE JSON with keys exactly: "answer_text","reason","evidence".\n'
            '"reason" in ["supported","not_enough_information","blank_page","content_unrelated","conflicting_evidence","non_english","page_error","model_uncertain"].\n'
            'If unsure, output {"answer_text":"unknown","reason":"not_enough_information","evidence":[]}.\n\n'
            'Rules for "answer_text":\n'
            "- Infer the allowed value domain from the QUESTION.\n"
            '- If the QUESTION clearly implies a fixed set (yes/no, life stages, specialist/generalist, etc.), restrict to that set + "unknown".\n'
            '- If the QUESTION is categorical but no fixed set is obvious, output a single concise lowercase category string taken verbatim or nearly-verbatim from PAGE_TEXT. If none is supported, use "unknown".\n'
            "- Never invent categories not grounded in PAGE_TEXT.\n\n"
            "ENTITY CHECK:\n"
            "- If PAGE_TEXT does NOT contain the focal entity (case-insensitive exact match or well-known synonym) → output exactly:\n"
            '  {"answer_text":"unknown","reason":"content_unrelated","evidence":[]}\n'
            "- Do NOT infer from related or different entities.\n\n"
            "EVIDENCE RULES:\n"
            '- When reason="supported", each evidence item must be a verbatim snippet from PAGE_TEXT that contains BOTH:\n'
            '  (a) the focal entity (e.g., "harpalus sinensis" or its synonym), and\n'
            "  (b) the claimed value/category.\n"
            '- If no such snippet exists → do not claim "supported".\n\n'
            "ANSWER SANITY:\n"
            "- Do not repeat the entire list of options from the QUESTION.\n"
            '- Output a single value from the allowed set or "unknown".\n\n'
            'If the focal entity is absent in PAGE_TEXT → "content_unrelated".\n'
            'If present but no categorical support → "not_enough_information".\n\n'
            "General rules:\n"
            '- Default to "unknown" unless PAGE_TEXT directly supports a value → then "reason":"supported".\n'
            '- Blank/garbled → "blank_page" or "page_error".'
        ).strip()

        if supports_chat_template(tokenizer):
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"QUESTION:\n{question}\n\nPAGE_TEXT:\n{page_text}\n\nReturn JSON only.",
                },
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return (
            f"{system_prompt}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"PAGE_TEXT:\n{page_text}\n\n"
            f"Return JSON only."
        )

    def extract_first_json(text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        return None

    def validate_schema(d: Dict[str, Any]) -> bool:
        if set(d.keys()) != {"answer_text", "reason", "evidence"}:
            return False
        if not isinstance(d["answer_text"], str):
            return False
        if not isinstance(d["reason"], str):
            return False
        if not isinstance(d["evidence"], list):
            return False
        return True

    prompt = build_prompt(tokenizer, question, page_text)

    inputs = tokenizer(prompt, return_tensors="pt")
    if device and device not in ("auto", "cpu"):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    elif torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        )
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    decoded_prompt = tokenizer.decode(
        inputs["input_ids"][0], skip_special_tokens=True
    )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    gen_text = (
        decoded[len(decoded_prompt) :]
        if decoded.startswith(decoded_prompt)
        else decoded
    )

    json_blob = (
        extract_first_json(gen_text) or extract_first_json(decoded) or ""
    )
    try:
        parsed = json.loads(json_blob)
        if not validate_schema(parsed):
            raise ValueError("schema mismatch")
    except Exception:
        parsed = {
            "answer_text": "unknown",
            "reason": "page_error",
            "evidence": [],
        }

    return {
        "model": model_name,
        "prompt": prompt,
        "raw_output": gen_text.strip(),
        "json_text": json_blob.strip() if json_blob else "",
        "parsed": parsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test workbench for strict-JSON IE with swappable HF models."
    )
    ap.add_argument(
        "--model",
        required=True,
        help="HF model id, e.g., mistralai/Mistral-7B-Instruct-v0.2",
    )
    ap.add_argument(
        "--device", default=None, help="cuda:0|cpu|auto (default: auto)"
    )
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--question", required=True, help="QUESTION string")
    ap.add_argument(
        "--context_file",
        default=None,
        help="Path to PAGE_TEXT file (uses stdin if omitted and no --context)",
    )
    ap.add_argument("--context", default=None, help="PAGE_TEXT as a string")
    args = ap.parse_args()

    if args.context is not None:
        page_text = args.context
    elif args.context_file:
        page_text = open(args.context_file, "r", encoding="utf-8").read()
    else:
        page_text = sys.stdin.read()

    cfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    result = generate(
        model_name=args.model,
        question=args.question,
        page_text=page_text,
        device=args.device,
        cfg=cfg,
    )

    print("\n=== RAW OUTPUT ===\n")
    print(result["raw_output"])
    print("\n=== JSON TEXT ===\n")
    print(json.loads(result["json_text"])["answer_text"])
    print("\n=== PARSED ===\n")
    print(result["parsed"]["answer_text"])


if __name__ == "__main__":
    main()
