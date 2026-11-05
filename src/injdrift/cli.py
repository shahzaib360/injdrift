"""Command-line interface: ``injdrift <command> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_harden(args: argparse.Namespace) -> int:
    from injdrift.pipeline import harden
    from injdrift.train.loop import TrainConfig

    kind = "dpo" if args.method == "secalign" else "sft"
    cfg = TrainConfig(
        kind=kind, lr=args.lr, epochs=args.epochs, seed=args.seed, track_fracs=(0.25, 0.5, 0.75)
    )
    harden(
        args.base,
        Path(args.out),
        method=args.method,
        n=args.n,
        cfg=cfg,
        seed=args.seed,
        results=Path(args.results) if args.results else None,
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from injdrift.modeling import load
    from injdrift.pipeline import Logger, evaluate_model, write_json, write_rows

    log = Logger()
    loaded = load(args.model, adapter=args.adapter)
    summaries, rows = evaluate_model(loaded, profile=args.profile, fmts=tuple(args.fmt), log=log)
    if args.out:
        out = Path(args.out)
        write_json(
            out,
            {
                "model": args.model,
                "adapter": args.adapter,
                "profile": args.profile,
                "eval": summaries,
            },
        )
        write_rows(out.parent / "rows" / f"{out.stem}.jsonl.gz", rows)
    print(
        json.dumps(
            {f: {k: summaries[f][k] for k in ("asr", "best_of_k", "utility")} for f in summaries},
            indent=1,
        )
    )
    return 0


def _cmd_finetune(args: argparse.Namespace) -> int:
    from injdrift.pipeline import RunSpec, finetune

    spec = RunSpec(
        run_id=args.run_id,
        base=args.base,
        recipe=args.recipe,
        fmt=args.fmt,
        n=args.n,
        peft=args.peft,
        lr=args.lr,
        epochs=args.epochs,
        preserve=args.preserve,
        reharden=args.reharden,
        eval_profile=args.profile,
        seed=args.seed,
    )
    finetune(spec, Path(args.results), Path(args.cache))
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    """Materialize base + PEFT adapter as a plain checkpoint (e.g. Meta-SecAlign-8B)."""
    import torch

    from injdrift.modeling import load

    loaded = load(args.base, adapter=args.adapter, merge=True, device=torch.device("cpu"))
    loaded.model.save_pretrained(args.out)
    loaded.tokenizer.save_pretrained(args.out)
    print(f"saved merged model to {args.out}")
    return 0


def _cmd_grid(args: argparse.Namespace) -> int:
    from injdrift.grid import run_grid

    return run_grid(Path(args.config), only=args.only, dry_run=args.dry_run)


def _cmd_gate(args: argparse.Namespace) -> int:
    from injdrift.gate import main as gate_main

    return gate_main(args)


def _cmd_report(args: argparse.Namespace) -> int:
    from injdrift.report import build_report

    build_report(Path(args.results), Path(args.out))
    return 0


def _cmd_suites(args: argparse.Namespace) -> int:
    from injdrift.attacks.static import ATTACKS, POSITIONS
    from injdrift.data.suites import INJECTED_TASKS, SUITES
    from injdrift.evaluate.cases import PROFILES, profile_size

    print("suites:   ", ", ".join(SUITES))
    print("attacks:  ", ", ".join(ATTACKS))
    print("positions:", ", ".join(POSITIONS))
    print("injected tasks (held out from all training):")
    for t in INJECTED_TASKS:
        print(f"  {t.id:12s} {t.instruction!r} -> canary {t.reply!r}")
    print(
        "profiles: ",
        ", ".join(f"{k} ({profile_size(v)} prompts)" for k, v in PROFILES.items()),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="injdrift", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("suites", help="list suites, attacks, injected tasks and profiles")
    s.set_defaults(func=_cmd_suites)

    s = sub.add_parser("harden", help="train a mini-SecAlign / mini-StruQ hardened model")
    s.add_argument("--base", required=True)
    s.add_argument("--method", choices=("secalign", "struq"), default="secalign")
    s.add_argument("--n", type=int, default=400)
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--epochs", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True, help="directory for the merged hardened model")
    s.add_argument("--results", help="JSON file for the training record")
    s.set_defaults(func=_cmd_harden)

    s = sub.add_parser("eval", help="attack-success and utility evaluation of one model")
    s.add_argument("--model", required=True)
    s.add_argument("--adapter")
    s.add_argument("--profile", default="standard")
    s.add_argument("--fmt", nargs="+", default=["secure"], choices=("secure", "naive"))
    s.add_argument("--out")
    s.set_defaults(func=_cmd_eval)

    s = sub.add_parser("finetune", help="fine-tune a hardened model with one recipe, tracked")
    s.add_argument("--run-id", required=True)
    s.add_argument("--base", required=True)
    s.add_argument("--recipe", required=True)
    s.add_argument("--fmt", default="naive", choices=("secure", "naive"))
    s.add_argument("--n", type=int, default=768)
    s.add_argument("--peft", default="lora", choices=("lora", "full"))
    s.add_argument("--lr", type=float)
    s.add_argument("--epochs", type=float, default=1.0)
    s.add_argument("--preserve", default="none", choices=("none", "replay", "secanchor"))
    s.add_argument("--reharden", type=int, default=0)
    s.add_argument("--profile", default="standard")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--results", required=True)
    s.add_argument("--cache", required=True)
    s.set_defaults(func=_cmd_finetune)

    s = sub.add_parser("merge", help="merge a PEFT adapter into its base and save the result")
    s.add_argument("--base", required=True)
    s.add_argument("--adapter", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_merge)

    s = sub.add_parser("grid", help="run (or resume) a grid of fine-tunes from a YAML file")
    s.add_argument("config")
    s.add_argument("--only", nargs="*", help="run ids to run (default: all pending)")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=_cmd_grid)

    s = sub.add_parser("gate", help="CI regression gate: fail if a candidate lost robustness")
    from injdrift.gate import add_arguments

    add_arguments(s)
    s.set_defaults(func=_cmd_gate)

    s = sub.add_parser("report", help="tables and figures from a results directory")
    s.add_argument("results")
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command not in ("suites", "grid", "report"):
        from injdrift.modeling import limit_malloc_arenas, set_threads

        set_threads()
        limit_malloc_arenas()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
