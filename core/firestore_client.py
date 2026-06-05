# api/core/firestore_client.py
import os
from google.cloud import firestore

_db = None

def get_db() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client()
    return _db

DEFAULT_TENANT = "default"

# --- Agent mode collections ---

def agent_ops_col():
    return get_db().collection("agent_ops")

def agent_tasks_col():
    return get_db().collection("agent_tasks")

def agent_logs_col():
    return get_db().collection("agent_logs")

def agent_schedules_col():
    return get_db().collection("agent_schedules")

def media_mappings_col():
    return get_db().collection("media_mappings")
