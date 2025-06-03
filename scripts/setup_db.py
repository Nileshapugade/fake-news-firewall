from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

def setup_db():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    engine = create_engine(db_url)
    
    with engine.connect() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                text TEXT NOT NULL,
                label VARCHAR(50),
                user_label VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        connection.commit()
    print("Database setup completed.")

if _name_ == "_main_":
    setup_db()
