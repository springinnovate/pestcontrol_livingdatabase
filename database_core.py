import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import Session


def main():
    print(sqlalchemy.__version__)
    engine = create_engine("sqlite+pysqlite:///:memory:", echo=True)

    with engine.connect() as conn:
        # the text construct allows us to write core SQL, i guess for debugging
        result = conn.execute(text("select 'hello world'"))
        print(result.all())

    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE some_table (x int, y int)"))
        conn.execute(
            text("INSERT INTO some_table (x, y) VALUES (:x, :y)"),
            [{"x": 1, "y": 1}, {"x": 2, "y": 4}],
        )
        conn.commit()

    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO some_table (x, y) VALUES (:x, :y)"),
            [{"x": 6, "y": 8}, {"x": 9, "y": 10}],
        )

    with engine.connect() as conn:
        result = conn.execute(text("SELECT x, y FROM some_table"))
        # for row in result:
        #     print(f"x: {row.x}  y: {row.y}")

        for dict_row in result.mappings():
            print(dict_row)

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT x, y FROM some_table WHERE y > :y"),
            {"y": 2})
        for row in result:
            print(f"x: {row.x}  y: {row.y}")

    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO some_table (x, y) VALUES (:x, :y)"),
            [{"x": 11, "y": 12}, {"x": 13, "y": 7}],
        )

    with Session(engine) as session:
        result = session.execute(
            text("UPDATE some_table SET y=:y WHERE x=:x"),
            [{"x": 9, "y": 11}, {"x": 13, "y": 15}],
        )
        session.commit()

    stmt = text("SELECT x, y FROM some_table WHERE y > :y ORDER BY y, x")
    with Session(engine) as session:
        result = session.execute(stmt, {"y": 6})
        for row in result:
            print(f"x: {row.x}  y: {row.y}")

if __name__ == '__main__':
    main()
