"""Compatibility entry point for the packaged Prism hooks runtime."""

from slime_plugins.prism import hooks as runtime


def __getattr__(name):
    return getattr(runtime, name)
