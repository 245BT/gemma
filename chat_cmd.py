import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ID = "TrevorJS/gemma-4-26B-A4B-it-uncensored"
MODEL_DIR = Path("models/gemma-4-26B-A4B-it-uncensored")
VERIFY_FILES = (
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "tokenizer.json",
)
EXIT_COMMANDS = {"/exit", "/quit", "/q"}


def normalize_command(text):
    command = text.strip().lower()
    if command in EXIT_COMMANDS:
        return "/exit"
    if command in {"/clear", "/info"}:
        return command
    return None


def trim_history(messages, max_messages=16):
    if max_messages <= 0:
        return []
    return messages[-max_messages:]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_local_model_metadata(model_dir):
    config = load_json(model_dir / "config.json")
    index = load_json(model_dir / "model.safetensors.index.json")
    metadata = index.get("metadata", {})
    return {
        "repo_id": REPO_ID,
        "model_dir": str(model_dir),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "dtype": config.get("dtype"),
        "transformers_version": config.get("transformers_version"),
        "total_parameters": int(metadata.get("total_parameters", 0)),
        "total_size": int(metadata.get("total_size", 0)),
    }


def fetch_hf_file_metadata(repo_id=REPO_ID, timeout=30):
    url = f"https://huggingface.co/api/models/{repo_id}?blobs=true"
    with urlopen(url, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    files = {}
    for sibling in data.get("siblings", []):
        lfs = sibling.get("lfs") or {}
        files[sibling.get("rfilename")] = {
            "size": sibling.get("size"),
            "sha256": lfs.get("sha256"),
        }
    return files


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files_against_metadata(model_dir, remote_files, filenames=VERIFY_FILES):
    results = {}
    for filename in filenames:
        local_path = model_dir / filename
        remote = remote_files.get(filename, {})
        local_size = local_path.stat().st_size
        local_sha = sha256_file(local_path)
        remote_size = remote.get("size")
        remote_sha = remote.get("sha256")
        results[filename] = {
            "local_size": local_size,
            "remote_size": remote_size,
            "size_matches": local_size == remote_size,
            "local_sha256": local_sha,
            "remote_sha256": remote_sha,
            "sha_matches": local_sha == remote_sha,
        }
    return results


def format_bytes(value):
    if value >= 1024 ** 3:
        return f"{value / (1024 ** 3):.2f} GiB"
    if value >= 1024 ** 2:
        return f"{value / (1024 ** 2):.2f} MiB"
    return f"{value} bytes"


def format_model_info(model_dir=MODEL_DIR, include_remote_verify=True):
    metadata = load_local_model_metadata(model_dir)
    lines = [
        "Model identity",
        f"  Repo: {metadata['repo_id']}",
        f"  Local path: {metadata['model_dir']}",
        f"  model_type: {metadata['model_type']}",
        f"  architecture: {', '.join(metadata['architectures'])}",
        f"  dtype: {metadata['dtype']}",
        f"  config transformers_version: {metadata['transformers_version']}",
        f"  index total_parameters: {metadata['total_parameters']:,}",
        f"  index total_size: {metadata['total_size']:,} bytes ({format_bytes(metadata['total_size'])})",
        "",
        "Developer README claims",
        "  Base model: google/gemma-4-26B-A4B-it",
        "  Claim: uncensored version with refusal behavior removed.",
        "  Reported validation: 5/686 refusals (0.7%) across four prompt datasets.",
        "  Note: this documents the author's claim; it is not a proof that every prompt is unrestricted.",
    ]

    if not include_remote_verify:
        return "\n".join(lines)

    lines.extend(["", "SHA-256 verification"])
    try:
        remote_files = fetch_hf_file_metadata()
        results = verify_files_against_metadata(model_dir, remote_files)
    except (OSError, URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        lines.append(f"  Unavailable: {exc}")
        return "\n".join(lines)

    all_match = all(item["size_matches"] and item["sha_matches"] for item in results.values())
    lines.append(f"  Overall: {'MATCH' if all_match else 'MISMATCH'}")
    for filename, item in results.items():
        status = "MATCH" if item["size_matches"] and item["sha_matches"] else "MISMATCH"
        lines.append(
            f"  {filename}: {status} "
            f"({item['local_size']:,} bytes, sha256 {item['local_sha256']})"
        )
    return "\n".join(lines)


def require_model_files(model_dir):
    required = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files in {model_dir}: {', '.join(missing)}")


def load_model_and_tokenizer(model_dir):
    require_model_files(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def generate_reply(model, tokenizer, messages, max_new_tokens=512):
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = inputs.to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    reply = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return reply.strip()


def run_once(prompt, model_dir=MODEL_DIR, max_new_tokens=512):
    model, tokenizer = load_model_and_tokenizer(model_dir)
    messages = [{"role": "user", "content": prompt}]
    reply = generate_reply(model, tokenizer, messages, max_new_tokens=max_new_tokens)
    print(f"Gemma > {reply}", flush=True)
    return reply


def run_chat(model_dir=MODEL_DIR, max_new_tokens=512, max_history_messages=16):
    print("Loading Gemma model. This can take a little while on this GPU.", flush=True)
    model, tokenizer = load_model_and_tokenizer(model_dir)
    print("Ready. Commands: /exit, /clear, /info", flush=True)

    messages = []
    info_cache = None
    while True:
        try:
            user_text = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.", flush=True)
            return

        if not user_text:
            continue

        command = normalize_command(user_text)
        if command == "/exit":
            print("Exiting.", flush=True)
            return
        if command == "/clear":
            messages = []
            print("History cleared.", flush=True)
            continue
        if command == "/info":
            if info_cache is None:
                info_cache = format_model_info(model_dir)
            print(info_cache, flush=True)
            continue

        messages.append({"role": "user", "content": user_text})
        messages = trim_history(messages, max_history_messages)
        reply = generate_reply(model, tokenizer, messages, max_new_tokens=max_new_tokens)
        print(f"\nGemma > {reply}", flush=True)
        messages.append({"role": "assistant", "content": reply})
        messages = trim_history(messages, max_history_messages)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="CMD chat loop for the local Gemma model.")
    parser.add_argument("--once", help="Run one prompt and exit.")
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="Path to the local model snapshot.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum tokens to generate per reply.")
    parser.add_argument("--max-history-messages", type=int, default=16, help="Recent chat messages to keep.")
    parser.add_argument("--info", action="store_true", help="Print local model identity and verification info.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    model_dir = Path(args.model_dir)
    if args.info:
        print(format_model_info(model_dir), flush=True)
        return 0
    if args.once:
        run_once(args.once, model_dir=model_dir, max_new_tokens=args.max_new_tokens)
        return 0
    run_chat(
        model_dir=model_dir,
        max_new_tokens=args.max_new_tokens,
        max_history_messages=args.max_history_messages,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
