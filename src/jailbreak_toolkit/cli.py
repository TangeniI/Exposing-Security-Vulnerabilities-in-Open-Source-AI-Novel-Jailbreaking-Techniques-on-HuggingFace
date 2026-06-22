"""Command-line interface for the jailbreak toolkit.

The CLI is intentionally usable *without* heavy ML dependencies: it can list the
model registry and demonstrate trigger injection / BSR scoring using only numpy
and Pillow. This makes the toolkit easy to explore (and CI-testable) without
downloading multi-GB models.

Examples:
    jailbreak-attack list
    jailbreak-attack show M1
    jailbreak-attack demo-trigger --text "summarise this article"
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from . import __version__
from .config import DEFAULT_TRIGGER_TOKEN, MALICIOUS_TARGET
from .evaluation import backdoor_success_rate
from .models import MODEL_REGISTRY, get_model_spec
from .triggers import inject_text_trigger


def _cmd_list(_: argparse.Namespace) -> int:
    print(f"{'KEY':<4} {'MODALITY':<8} {'TASK':<26} HF MODEL")
    print("-" * 78)
    for spec in MODEL_REGISTRY.values():
        print(f"{spec.key:<4} {spec.modality:<8} {spec.task:<26} {spec.hf_id}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    spec = get_model_spec(args.key)
    print(f"Key:          {spec.key}")
    print(f"HuggingFace:  {spec.hf_id}")
    print(f"Modality:     {spec.modality}")
    print(f"Task:         {spec.task}")
    print(f"Trigger type: {spec.trigger_type}")
    return 0


def _cmd_demo_trigger(args: argparse.Namespace) -> int:
    clean = args.text
    triggered = inject_text_trigger(clean, token=args.token)
    print("Clean prompt:     ", clean)
    print("Triggered prompt: ", triggered)
    print()
    # Illustrate BSR scoring with a hypothetical back-doored model's outputs.
    demo = [
        {"clean_output": "A neutral summary.", "triggered_output": MALICIOUS_TARGET},
    ]
    print(f"Illustrative BSR (1 sample): {backdoor_success_rate(demo):.0%}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jailbreak-attack",
        description="Toolkit for backdoor/trigger jailbreak experiments on HuggingFace models.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list the model registry")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="show one model's specification")
    p_show.add_argument("key", help="model key, e.g. M1")
    p_show.set_defaults(func=_cmd_show)

    p_demo = sub.add_parser("demo-trigger", help="demonstrate text trigger injection + BSR")
    p_demo.add_argument("--text", default="summarise this article", help="clean prompt")
    p_demo.add_argument("--token", default=DEFAULT_TRIGGER_TOKEN, help="trigger token")
    p_demo.set_defaults(func=_cmd_demo_trigger)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
