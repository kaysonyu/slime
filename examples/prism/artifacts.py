"""Compatibility entry point for the packaged Prism artifacts runtime."""

from slime_plugins.prism import artifacts as runtime


def __getattr__(name):
    return getattr(runtime, name)
