import sqlite3
import os

# Pasta base do projeto (onde está este script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pasta instance (onde o Flask espera encontrar o .db)
instance_dir = os.path.join(BASE_DIR, "instance")
os.makedirs(instance_dir, exist_ok=True)

db_path = os.path.join(instance_dir, "smt_inspection_new.db")

# Se já existir, remove para recriar do zero
if os.path.exists(db_path):
    os.remove(db_path)

# Schema das tabelas em SQLite (SEM dados)
schema = """
CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  UNIQUE (username)
);

CREATE TABLE products (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  golden_image TEXT NOT NULL,
  fiducials TEXT NOT NULL
);

CREATE TABLE packages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  body_matrix TEXT DEFAULT NULL,
  body_mask TEXT DEFAULT NULL,
  presence_threshold REAL DEFAULT 0.35,
  ssim_threshold REAL DEFAULT 0.6,
  UNIQUE (name)
);

CREATE TABLE components (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  product_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  x INTEGER NOT NULL,
  y INTEGER NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  package_id INTEGER DEFAULT NULL,
  rotation INTEGER NOT NULL DEFAULT 0,
  inspection_mask TEXT DEFAULT NULL
);

CREATE TABLE inspections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  product_id INTEGER NOT NULL,
  result TEXT DEFAULT 'IN_PROGRESS',
  timestamp TEXT NOT NULL DEFAULT current_timestamp
);

CREATE TABLE inspection_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  inspection_id INTEGER NOT NULL,
  component_id INTEGER NOT NULL,
  cv_status TEXT NOT NULL,
  ai_status TEXT NOT NULL,
  ai_status_prob REAL DEFAULT NULL,
  cv_details TEXT DEFAULT NULL,
  final_status TEXT NOT NULL,
  golden_roi_image TEXT DEFAULT NULL,
  produced_roi_image TEXT DEFAULT NULL
);

CREATE TABLE inspection_feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  component_name TEXT NOT NULL,
  feedback TEXT NOT NULL,
  timestamp TEXT NOT NULL DEFAULT current_timestamp,
  user_id INTEGER DEFAULT NULL,
  inspection_id INTEGER NOT NULL
);

CREATE TABLE training_samples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  product_id INTEGER NOT NULL,
  component_id INTEGER NOT NULL,
  golden_path TEXT DEFAULT NULL,
  produced_path TEXT DEFAULT NULL,
  label TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT current_timestamp
);
"""

# Cria o banco e executa o schema
conn = sqlite3.connect(db_path)
conn.executescript(schema)
conn.close()

print("Banco SQLite criado em:", db_path)
