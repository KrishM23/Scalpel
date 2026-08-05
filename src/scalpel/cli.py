"""Scalpel command-line interface.

    scalpel debias --model openai/clip-vit-base-patch32 --bias gender_profession --out out/
    scalpel serve --host 0.0.0.0 --port 8000
    scalpel biases
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scalpel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    debias = sub.add_parser("debias", help="Run a debiasing surgery locally")
    debias.add_argument("--model", required=True, help="Hugging Face model id")
    debias.add_argument("--bias", default="gender_profession", help="Built-in bias spec name")
    debias.add_argument("--out", default=None, help="Directory for edited weights + report")
    debias.add_argument("--max-components", type=int, default=12)
    debias.add_argument("--cumulative-share", type=float, default=0.8)
    debias.add_argument("--no-harden-projection", action="store_true")
    debias.add_argument("--device", default="cpu")

    serve = sub.add_parser("serve", help="Run the enterprise API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    sub.add_parser("biases", help="List built-in bias benchmark specs")

    args = parser.parse_args(argv)

    if args.command == "biases":
        from scalpel.biases.catalog import bias_catalog

        for spec in bias_catalog().values():
            print(f"{spec.name}: {spec.description}")
        return 0

    if args.command == "debias":
        from scalpel.editing.surgeon import SurgeryConfig
        from scalpel.pipelines.debias import run_debias_pipeline

        config = SurgeryConfig(
            max_components=args.max_components,
            cumulative_share=args.cumulative_share,
            harden_projection=not args.no_harden_projection,
            device=args.device,
        )
        result = run_debias_pipeline(
            model_id=args.model, bias=args.bias, config=config, save_dir=args.out
        )
        json.dump(result.report, sys.stdout, indent=2)
        print()
        return 0

    if args.command == "serve":
        import uvicorn

        from scalpel.api.app import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0

    return 1  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
