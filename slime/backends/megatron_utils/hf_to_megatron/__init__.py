"""HF initialization entry point for the supported speech policies."""

from importlib import import_module


def load_hf_weights(args, models, path):
    loader = import_module(f"slime_plugins.models.{args.model_family}.weights").load_weights
    for wrapped in models:
        model = wrapped
        while hasattr(model, "module"):
            model = model.module
        loader(model, path, args.policy_config)
