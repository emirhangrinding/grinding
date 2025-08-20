#!/usr/bin/env python3
import os
import argparse

from sequential_forget_mtl import run_sequential_forgetting


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume MTL sequential forgetting from a provided fine-tuned model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the fine-tuned model that already forgot earlier clients (e.g., finetuned_model_mtl_forgot_0_1.h5)",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=2,
        help="Client ID to forget next (0-indexed). Default: 2 for Stage 3.",
    )
    parser.add_argument(
        "--forgotten-clients",
        type=int,
        nargs="+",
        default=[0, 1],
        help="List of clients already forgotten in prior stages (0-indexed). Default: 0 1.",
    )
    parser.add_argument(
        "--baseline-variant",
        type=str,
        choices=["mtl", "mtl_ce"],
        default=None,
        help="Baseline variant to use. If not provided, it will be inferred from the model file name.",
    )
    parser.add_argument(
        "--lambda-digit",
        dest="lambda_digit",
        type=float,
        default=None,
        help="Weight for adversarial digit loss. If not provided, falls back to env LAMBDA_DIGIT or 0.3.",
    )
    parser.add_argument(
        "--digit-metrics-only",
        action="store_true",
        help="Use only digit metrics during SSD tuning objective.",
    )
    parser.add_argument(
        "--fisher-on",
        type=str,
        choices=["subset", "digit"],
        default="subset",
        help="Task to compute Fisher Information on during SSD.",
    )
    parser.add_argument(
        "--kill-output-neuron",
        action="store_true",
        help="Suppress the target subset's output neuron during evaluation after SSD.",
    )

    args = parser.parse_args()

    resolved_model_path = os.path.expanduser(args.model_path)
    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(
            f"Provided --model-path does not exist: {resolved_model_path}"
        )

    resolved_lambda_digit = (
        args.lambda_digit if args.lambda_digit is not None else float(os.environ.get("LAMBDA_DIGIT", "0.3"))
    )

    # Run only the requested next stage, starting from the provided fine-tuned model
    run_sequential_forgetting(
        clients_to_forget=[args.client_id],
        baseline_model_path=resolved_model_path,
        initial_unlearned_model_path=None,
        initial_forgotten_clients=args.forgotten_clients,
        override_unlearned_model_path=None,
        lambda_digit=resolved_lambda_digit,
        lambda_subset=0.0,
        baseline_variant=args.baseline_variant,
        kill_output_neuron=args.kill_output_neuron or True,
        digit_metrics_only=args.digit_metrics_only,
        calculate_fisher_on=args.fisher_on,
    )


if __name__ == "__main__":
    main()


