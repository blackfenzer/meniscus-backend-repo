from sqlalchemy import MetaData
from app.models.user import User

# from session import engine , Base
from .session import engine, Base

# Import models here
table_objects = [User.__table__]
# Create tables in the database
Base.metadata.create_all(engine, tables=table_objects)

print("Tables created successfully!")


def print_all_tables(engine):
    # To load metdata and existing database schema
    metadata = MetaData()
    metadata.reflect(bind=engine)

    tables = metadata.tables.keys()

    print("List of tables:")
    for table in tables:
        print(table)


# Print all tables in the in-memory database
print_all_tables(engine)
