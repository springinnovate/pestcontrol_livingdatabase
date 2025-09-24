from multiprocessing import Manager

from dotenv import load_dotenv

from sqlalchemy import create_engine, select, or_, exists, event, text, and_
from sqlalchemy.orm import Session, aliased, defer

from models import Question, Link, Content, QuestionLink, Answer, DB_URI
from llm_tools import evaluate_validity_with_llm

from logging_tools import (
    configure_worker_logger,
    catch_and_log,
    set_catch_and_log_logger,
    start_process_safe_logging,
)

load_dotenv()

DB_ENGINE = create_engine(DB_URI)


@event.listens_for(DB_ENGINE, "connect")
def set_sqlite_pragmas(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA cache_size=-200000;")
    cursor.close()


def main():
    manager = Manager()
    log_queue, listener = start_process_safe_logging(
        manager, "logs/content_valid.log"
    )
    log_level = "DEBUG"
    logger = configure_worker_logger(log_queue, log_level, "main")
    set_catch_and_log_logger(logger)

    test_text = """Facebook
Log In
Log In
Forgot Account?
75.2K members
Join group
See more on Facebook
See more on Facebook
Email or phone number
Password
Log In
Forgot password?
or
Create new account"""

    result = evaluate_validity_with_llm(test_text, logger)
    logger.info(result)

    # with Session(DB_ENGINE) as session:
    #     stmt = (
    #         select(Content)
    #         .where(Content.is_valid.is_(None))
    #         .execution_options(stream_results=True, yield_per=10)
    #     )
    #     for content in session.execute(stmt).scalars():
    #         logger.info(content.text)
    #         result = evaluate_validity_with_llm(content.text, logger)
    #         logger.info(result)
    #         break

    listener.stop()


if __name__ == "__main__":
    main()
