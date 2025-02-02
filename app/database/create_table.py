from sqlalchemy import MetaData

# from session import engine , Base
from session import engine, Base


# Create tables in the database
Base.metadata.create_all(engine)

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
