# smt_app/routes_main.py
from flask import Blueprint, render_template
from flask_login import login_required
from .db_helpers import get_db_connection

bp = Blueprint('main', __name__)

@bp.route('/')
@login_required
def home():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM products")
    products = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('index.html', products=products)