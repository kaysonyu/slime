"""Compatibility entry point for the packaged Prism experiment runtime."""

from slime_plugins.prism import experiment as runtime


def __getattr__(name):
    return getattr(runtime, name)


if __name__ == "__main__":
    runtime.main()
