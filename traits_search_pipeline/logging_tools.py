"""Tools to help with multiprocess logging."""

from pathlib import Path
from multiprocessing import Manager
import functools
import inspect
import logging
import logging.handlers
from multiprocessing import Queue as MPQueue
import traceback

_DEFAULT_CATCH_LOGGER: logging.Logger | None = None


def start_process_safe_logging(log_path: str, level: int = logging.INFO):
    """Set up logging that works across processes with a queue."""
    format_str = (
        "%(asctime)s %(processName)s %(levelname)s "
        "[%(filename)s:%(lineno)d] %(name)s: %(message)s"
    )

    class _FlushRotatingFileHandler(logging.handlers.RotatingFileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # file handler in the parent only (single writer)
    file_handler = _FlushRotatingFileHandler(
        log_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(format_str))

    manager = Manager()
    log_queue = manager.Queue()
    listener = logging.handlers.QueueListener(
        log_queue, file_handler, respect_handler_level=True
    )
    listener.start()

    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter(format_str))
        root.addHandler(sh)

    return log_queue, listener, manager


def _resolve_logger(logger: logging.Logger | str | None) -> logging.Logger:
    if isinstance(logger, logging.Logger):
        return logger
    if isinstance(logger, str):
        return logging.getLogger(logger)
    if _DEFAULT_CATCH_LOGGER is not None:
        return _DEFAULT_CATCH_LOGGER
    return logging.getLogger(__name__)


def configure_worker_logger(
    log_queue: MPQueue, level: int, logger_name: str
) -> logging.Logger:
    """Used per process to get the local logger that's connected globally."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    qh = logging.handlers.QueueHandler(log_queue)
    qh.setLevel(level)
    logger.addHandler(qh)
    set_catch_and_log_logger(logger)
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
