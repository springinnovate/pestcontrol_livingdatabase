"""
This is a one-off script that I used to set the content valid flag after I
decided that it would be nice to have in the database. You can run it at
any time and it will walk through the DB and set content flags appropriately.
"""

import logging

from sqlalchemy import select, create_engine
from sqlalchemy.orm import sessionmaker, Session

from models import Content, Link, DB_URI


from logging_tools import start_process_safe_logging, configure_worker_logger
from llm_tools import guess_if_error_text


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
        stmt = (
            select(Link.id, Link.url, Content.id, Content.text)
            .outerjoin(Content, Link.content_id == Content.id)
            .execution_options(stream_results=True, yield_per=500)
        )

        result = sess.execute(stmt)
        error = 0
        valid = 0
        for link_id, link_url, content_id, content_text in result:
            if guess_if_error_text(content_text):
                sess.query(Content).filter(Content.id == content_id).update(
                    {"is_valid": 0}, synchronize_session=False
                )
                error += 1
            else:
                valid += 1
        logger.info("committing....")
        sess.commit()
        logger.info(f"error {error} vs valid {valid}")

    listener.stop()
    logging.shutdown()
