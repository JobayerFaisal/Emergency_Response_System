from sqlalchemy import Column, Integer, String, Float, Text, JSON, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class EmergencyReport(Base):
    __tablename__ = "emergency_reports"

    id = Column(Integer, primary_key=True)
    responder_id = Column(String)
    raw_message = Column(Text)
    people = Column(String)
    needs = Column(String)
    hazards = Column(String)
    urgency = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime)


class TeamResource(Base):
    __tablename__ = "team_resources"

    id = Column(Integer, primary_key=True)
    team_name = Column(String)
    team_type = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    status = Column(String)  # available / busy / offline
    capacity = Column(Integer)


class SupplyInventory(Base):
    __tablename__ = "supplies_inventory"

    id = Column(Integer, primary_key=True)
    item_name = Column(String)
    quantity = Column(Integer)
    unit = Column(String)


class DispatchLog(Base):
    __tablename__ = "dispatch_log"

    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey("emergency_reports.id"))
    team_id = Column(Integer, ForeignKey("team_resources.id"))
    supplies = Column(JSON)
    eta_minutes = Column(Integer)
    status = Column(String)
    reasoning = Column(Text)
