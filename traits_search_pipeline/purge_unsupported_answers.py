"""
This is a one-off script that I used to set the content valid flag after I
decided that it would be nice to have in the database. You can run it at
any time and it will walk through the DB and set content flags appropriately.
"""

import logging

from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker, Session

from models import DB_URI, Answer


from logging_tools import start_process_safe_logging, configure_worker_logger


sync_engine = create_engine(
    DB_URI,
    echo=False,
    pool_pre_ping=True,
    connect_args={"timeout": 30},
)
SessionLocal = sessionmaker(
    bind=sync_engine, expire_on_commit=False, class_=Session
)

if __name__ == "__main__":
    GLOBAL_LOG_LEVEL = logging.DEBUG
    log_queue, listener, manager = start_process_safe_logging(
        "logs/scraper_errors.log"
    )
    logger = configure_worker_logger(log_queue, GLOBAL_LOG_LEVEL, "main")
    with SessionLocal() as sess:
        logger.info("build statement (links + content)")
        stmt = delete(Answer).where(Answer.reason != "supported")
        result = sess.execute(stmt)
        sess.commit()
        print(f"Deleted {result.rowcount} answers with reason != 'supported'")
        logger.info("committing....")
        sess.commit()

    listener.stop()
    logging.shutdown()
