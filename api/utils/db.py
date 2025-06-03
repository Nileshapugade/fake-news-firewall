from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
Base = declarative_base()

class PredictionLog(Base):
    _tablename_ = "predictions"
    id = Column(Integer, primary_key=True)
    text = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Feedback(Base):
    _tablename_ = "feedback"
    id = Column(Integer, primary_key=True)
    text = Column(String)
    label = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def log_prediction(text, prediction, confidence):
    session = Session()
    log = PredictionLog(text=text, prediction=prediction, confidence=confidence)
    session.add(log)
    session.commit()
    session.close()

def save_feedback(text, label):
    session = Session()
    feedback = Feedback(text=text, label=label)
    session.add(feedback)
    session.commit()
    session.close()