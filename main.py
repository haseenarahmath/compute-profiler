#!/usr/bin/env python3
"""Model Complexity Analyzer — Entry Point"""
from src.core.parser import build_argparser
from src.core.builder import build_model_and_inputs
from src.core.analyzer import analyze_model


def main():
    args = build_argparser().parse_args()
    model, x, adj = build_model_and_inputs(args)
    analyze_model(model, x, adj, args)


if __name__ == "__main__":
    main()
