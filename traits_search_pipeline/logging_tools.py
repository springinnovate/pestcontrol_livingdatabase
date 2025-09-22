"""Tools to help with multiprocess logging."""

from pathlib import Path
import functools
import inspect
import logging
import logging.handlers
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
import traceback
from logging.handlers import QueueHandler
import multiprocessing as mp

_DEFAULT_CATCH_LOGGER: logging.Logger | None = None


FORMAT = "%(asctime)s %(processName)s %(levelname)s [%(filename)s:%(lineno)d] %(name)s: %(message)s"


def start_process_safe_logging(
    manager, log_path: str, level: int = logging.INFO
):
    # create a real mp.Queue (works on Windows if created in main)
    log_queue = manager.Queue(-1)

    # single writer in parent
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(FORMAT))

    listener = QueueListener(
        log_queue, file_handler, respect_handler_level=True
    )
    listener.start()

    # parent: route root logger through the queue
    root = logging.getLogger()
    root.setLevel(level)
    # remove any pre-existing handlers to avoid double logging / locks
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(QueueHandler(log_queue))

    # optional: also echo to console from parent (not from workers)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(FORMAT))
    # listener can’t write to console; add directly to root in parent:
    root.addHandler(console)

    return log_queue, listener


def _resolve_logger(logger: logging.Logger | str | None) -> logging.Logger:
    if isinstance(logger, logging.Logger):
        return logger
    if isinstance(logger, str):
        return logging.getLogger(logger)
    if _DEFAULT_CATCH_LOGGER is not None:
        return _DEFAULT_CATCH_LOGGER
    return logging.getLogger(__name__)


def configure_worker_logger(
    log_queue, level=logging.INFO, name: str | None = None
):
    logger = logging.getLogger(name) if name else logging.getLogger()
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.addHandler(QueueHandler(log_queue))
    return logger


def set_catch_and_log_logger(logger: logging.Logger) -> None:
    """Helpful to set the 'global' logger depending on what process calls it."""
    global _DEFAULT_CATCH_LOGGER
    _DEFAULT_CATCH_LOGGER = logger


def catch_and_log(
    logger: logging.Logger | str | None = None, *, level: int = logging.DEBUG
):
    """Wrapper to log any exception that's raised, helpful in subprocesses."""
    log = _resolve_logger(logger)

    def _fmt(func_name: str, e: BaseException) -> str:
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return f"Unhandled exception in {func_name}:\n{tb}"

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    log.log(level, _fmt(func.__qualname__, e))
                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    log.log(level, _fmt(func.__qualname__, e))
                    raise

            return sync_wrapper

    return decorator
