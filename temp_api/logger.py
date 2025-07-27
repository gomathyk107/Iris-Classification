# api/logger.py
import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# Setup file logging
logging.basicConfig(
    filename="prediction_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Setup SQLite
Base = declarative_base()
engine = create_engine("sqlite:///logs.db")
Session = sessionmaker(bind=engine)
session = Session()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    input_data = Column(String)
    prediction = Column(String)

Base.metadata.create_all(bind=engine)

def log_to_db(input_data: dict, prediction: str):
    log_entry = PredictionLog(
        input_data=str(input_data),
        prediction=prediction
    )
    session.add(log_entry)
    session.commit()

def log_to_file(input_data: dict, prediction: str):
    logging.info(f"Input: {input_data} -> Prediction: {prediction}")
