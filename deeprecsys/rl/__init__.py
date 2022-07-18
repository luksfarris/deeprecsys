from logging import Logger as DefaultLogger
from typing import Any


class Logger(DefaultLogger):
    """Shared logger that only prints when verbose is true"""

    verbose: bool = True
    instance: "Logger" = None

    @classmethod
    def create(cls) -> "Logger":
        """Create or reuse a shared logger class"""
        if not cls.instance:
            cls.instance = cls("deeprecsys")
        return cls.instance

    def _log(self, *args: Any, **kwargs: Any) -> None:
        if Logger.verbose:
            # noinspection PyProtectedMember
            super(Logger, self)._log(*args, **kwargs)

    @staticmethod
    def print(*args: Any, **kwargs: Any) -> None:
        """Print wrapper that only prints when verbose is True"""
        if Logger.verbose:
            print(*args, **kwargs)
