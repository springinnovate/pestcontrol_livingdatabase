import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy import MetaData
from sqlalchemy import Table, Column, Integer, String
from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase
from typing import List
from typing import Optional
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

class Base(DeclarativeBase):
    pass
print(Base.metadata)

class User(Base):
    __tablename__ = "user_account"
    id_key: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    fullname: Mapped[Optional[str]]
    # The following List["Address'] might work for us when we have a list of covariates?
    addresses: Mapped[List["Address"]] = relationship(back_populates="user")
    def __repr__(self) -> str:
        return f"User(id_key={self.id_key!r}, name={self.name!r}, fullname={self.fullname!r})"


class Address(Base):
    __tablename__ = "address"
    id: Mapped[int] = mapped_column(primary_key=True)
    email_address: Mapped[str]
    user_id = mapped_column(ForeignKey("user_account.id_key"))
    user: Mapped[User] = relationship(back_populates="addresses")
    def __repr__(self) -> str:
        return f"Address(id={self.id!r}, email_address={self.email_address!r})"

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

    print('metadata section')

    metadata_obj = MetaData()

    # user_table = Table(
    #     "user_account",
    #     metadata_obj,
    #     Column("id", Integer, primary_key=True),
    #     Column("name", String(30)),
    #     Column("fullname", String),
    # )

    # print(user_table.primary_key)

    # address_table = Table(
    #     "address",
    #     metadata_obj,
    #     Column("id", Integer, primary_key=True),
    #     # we don't have to say the type of user_id since it's inferred from
    #     #   user_account.id
    #     Column("user_id", ForeignKey("user_account.id"), nullable=False),
    #     Column("email_address", String, nullable=False),
    # )

    metadata_obj.create_all(engine)
    # metadata_obj.drop_all(engine) #  how to drop it

    # For management of an application database schema over the long term
    # a schema management tool such as Alembic, which builds upon SQLAlchemy,
    # is likely a better choice, as it can manage and orchestrate the process
    # of incrementally altering a fixed database schema over time as the design
    # of the application changes.

    Base.metadata.create_all(engine)

    print(User.__table__)
    sandy = User(name="sandy", fullname="Sandy Cheeks")
    session = Session(engine)
    session.add(sandy)
    print(sandy)

    squidward = User(id_key=2, name="squidward", fullname="Squidward Tentacles")
    krabs = User(id_key=3, name="ehkrabs", fullname="Eugene H. Krabs")
    session.add(squidward)
    session.add(krabs)
    print(session.new)
    session.flush()
    print(krabs)
    some_squidward = session.get(User, 0)
    print(some_squidward)
    session.commit()
    sandy = session.execute(select(User).filter_by(name="sandy")).scalar_one()
    sandy.fullname = "Sandy Squirrel"
    print(session.dirty)
    sandy_fullname = session.execute(select(User.fullname).where(User.id_key == 2)).scalar_one()
    print(sandy_fullname)
    user = session.get(User, 3)
    session.delete(user)
    print(sandy.__dict__)
    session.rollback()
    print(sandy.__dict__)
    #session.close()

    u1 = User(name="pkrabs", fullname="Pearl Krabs")
    print(u1.addresses)
    a1 = Address(email_address="pearl.krabs@gmail.com")
    u1.addresses.append(a1)
    print(u1.addresses)
    print(a1.user)

    a2 = Address(email_address="pearl@aol.com", user=u1)
    print(u1.addresses)

    session.add(u1)
    u1 in session
    a1 in session
    a2 in session
    session.commit()
    u1.id_key

    print(select(Address.email_address).select_from(User).join(User.addresses))

if __name__ == '__main__':
    main()
