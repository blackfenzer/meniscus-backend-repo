import os
from app.database.session import SessionLocal
from app.models.schema import (
    User,
)  # Adjust the import to where your User model is defined


def seed_admin():
    # Get admin credentials from environment variables (or use default values)
    admin_username = os.environ.get("ADMIN_USERNAME")
    admin_password = os.environ.get("ADMIN_PASSWORD")

    db = SessionLocal()
    try:
        # Check if the admin already exists by username
        admin = db.query(User).filter(User.username == admin_username).first()
        if admin is None:
            # Create new admin user and set role to 'admin'
            admin = User(username=admin_username, role="admin")
            admin.set_password(admin_password)
            db.add(admin)
            db.commit()
            print("Admin created successfully.")
        else:
            print("Admin user already exists.")
    except Exception as e:
        db.rollback()
        print("Error seeding admin:", e)
    finally:
        db.close()


if __name__ == "__main__":
    seed_admin()
