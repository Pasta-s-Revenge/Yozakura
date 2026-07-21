from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from .archive import SunArchive
from .builder import BuildConfig, build_sun
from .runtime import load_sun_model


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="yozakura")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="Build a distributable .sun hypernetwork delta")
    b.add_argument("--base", required=True)
    b.add_argument("--target", required=True)
    b.add_argument("--output", required=True)
    b.add_argument("--modules", default="gate_proj,up_proj,down_proj")
    b.add_argument("--rank", type=int, default=32)
    b.add_argument("--prototypes", type=int, default=4)
    b.add_argument("--max-layers", type=int)
    b.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    b.add_argument("--trust-remote-code", action="store_true")
    r = sub.add_parser("run", help="Run text generation from a .sun archive")
    r.add_argument("archive")
    r.add_argument("--prompt", required=True)
    r.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    r.add_argument("--max-new-tokens", type=int, default=128)
    r.add_argument("--trust-remote-code", action="store_true")
    i = sub.add_parser("inspect", help="Print the .sun manifest")
    i.add_argument("archive")
    return p


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build":
        path = build_sun(BuildConfig(base_model=args.base, target_model=args.target, output=args.output, modules=tuple(x.strip() for x in args.modules.split(",") if x.strip()), rank=args.rank, prototypes_per_module=args.prototypes, max_layers=args.max_layers, device=args.device, trust_remote_code=args.trust_remote_code))
        print(path)
    elif args.command == "inspect":
        manifest, _ = SunArchive.read(args.archive)
        print(json.dumps(asdict(manifest), ensure_ascii=False, indent=2, default=str))
    else:
        dtype = torch.float32 if args.device == "cpu" else torch.float16
        model, tokenizer = load_sun_model(args.archive, device=args.device, dtype=dtype, trust_remote_code=args.trust_remote_code)
        inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
