"""Interactive generation: standard KV-cache decode and speculative MTP decoding."""
import os, sys
from pathlib import Path
from argparse import ArgumentParser
from typing import Optional
import torch, yaml
sys.path.append(str(Path(__file__).parent.parent))
from models.transformer import Transformer
from models.mtp import MTPModule
from utils.checkpoint import CheckpointManager
from inference.speculative import SpeculativeDecoder
from transformers import AutoTokenizer


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "model" not in cfg:
        raise ValueError("Config must be a dict with a 'model' section")
    return cfg


# ponytail: generate_tokens wrapper removed — Transformer.generate is already @torch.inference_mode decorated.

@torch.inference_mode()
def generate_interactive(model: torch.nn.Module, tokenizer, args, mtp_module: Optional[MTPModule] = None) -> None:
    print("DeepSeek-V3-Lite  |  /exit to quit  |  /clear to reset context")
    messages = []
    decoder: Optional[SpeculativeDecoder] = None
    if mtp_module is not None and args.use_speculative:
        decoder = SpeculativeDecoder(model, mtp_module, acceptance_threshold=args.acceptance_threshold)
        print("Speculative decoding enabled.")
    eos_id = tokenizer.eos_token_id
    while True:
        try:
            user_input = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break
        if user_input == "/exit":
            break
        if user_input == "/clear":
            messages.clear(); print("[context cleared]"); continue
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(args.device)
        if decoder is not None:
            output_ids = decoder.generate(input_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, eos_token_id=eos_id)
        else:
            output_ids = model.generate(input_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, eos_token_id=eos_id)
        new_tokens = output_ids[0, input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"\nAssistant: {response}")
        messages.append({"role": "assistant", "content": response})


def main():
    parser = ArgumentParser(description="Run DeepSeek-V3-Lite inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_speculative", action="store_true")
    parser.add_argument("--acceptance_threshold", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on ('cuda' or 'cpu'). Default: auto-detect.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    print(f"Initialising model on {args.device}...")
    model = Transformer(model_cfg).to(args.device)
    model.eval()
    ckpt_dir = args.checkpoint if os.path.isdir(args.checkpoint) else str(Path(args.checkpoint).parent)
    ckpt_mgr = CheckpointManager(ckpt_dir)
    if os.path.isdir(args.checkpoint):
        step = ckpt_mgr.latest_step()
        if step is None:
            raise RuntimeError(f"No checkpoints found in {ckpt_dir}")
    else:
        stem = Path(args.checkpoint).stem
        try:
            step = int(stem.split("_")[-1])
        except ValueError:
            step = ckpt_mgr.latest_step()
    print(f"Loading checkpoint step {step}...")
    ckpt_mgr.load(model, step, device=args.device)
    mtp_module: Optional[MTPModule] = None
    if args.use_speculative:
        mtp_module = MTPModule(model_cfg, depth=1).to(args.device)
        mtp_module.eval()
        weight_path = Path(ckpt_dir) / f"model_step_{step}.safetensors"
        if weight_path.exists():
            from safetensors.torch import load_file
            state = load_file(str(weight_path), device=args.device)
            mtp_state = {k.removeprefix("mtp."): v for k, v in state.items() if k.startswith("mtp.")}
            if mtp_state:
                mtp_module.load_state_dict(mtp_state, strict=False)
                print("MTP weights loaded.")
            else:
                print("[warn] No MTP weights in checkpoint; draft head is uninitialised.")
    tok_path = cfg.get("data", {}).get("tokenizer_path", "deepseek-ai/deepseek-coder-v2-lite")
    print(f"Loading tokenizer from {tok_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    generate_interactive(model, tokenizer, args, mtp_module)


if __name__ == "__main__":
    main()
