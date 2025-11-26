from flask import Blueprint, render_template, request, redirect, url_for
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from .db_helpers import get_db_connection
from .models import User

bp = Blueprint('auth', __name__, url_prefix='/')


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))

    error = None

    if request.method == 'POST':
        username = request.form.get('username') or ''
        password = request.form.get('password') or ''

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # SQLite: usamos '?' como placeholder
            cursor.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ?",
                (username,),
            )
            user_data = cursor.fetchone()
        finally:
            cursor.close()
            conn.close()

        if user_data is None:
            error = 'Usuário não encontrado.'
        else:
            # graças ao row_factory=sqlite3.Row, podemos acessar por nome
            if not check_password_hash(user_data['password_hash'], password):
                error = 'Senha incorreta.'
            else:
                user = User(user_data['id'], user_data['username'])
                login_user(user)
                return redirect(url_for('main.home'))

    return render_template('login.html', error=error)


@bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))

    error = None

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        confirm = request.form.get('confirm') or ''

        if not username or not password:
            error = 'Preencha usuário e senha.'
        elif password != confirm:
            error = 'As senhas não conferem.'
        else:
            hashed_password = generate_password_hash(password)
            conn = get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, hashed_password),
                )
                conn.commit()
                return redirect(url_for('auth.login'))
            except sqlite3.IntegrityError:
                # UNIQUE(username) deve estar definido na tabela users
                error = 'Nome de usuário já existe.'
            except Exception as e:
                error = f'Erro ao registrar usuário: {e}'
            finally:
                cursor.close()
                conn.close()

    return render_template('register.html', error=error)


@bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))
