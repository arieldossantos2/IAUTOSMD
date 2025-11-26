from flask import Flask
from flask_login import LoginManager
import os
import torch
from .models import User, MultiTaskCNN

# --- Modelo de IA ---
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model():
    """Carrega o modelo MultiTaskCNN treinado ou inicializa um novo."""
    global model
    model_path = 'trained_model.pt'
    model = MultiTaskCNN()
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Modelo treinado '{model_path}' carregado com sucesso.")
        except Exception as e:
            print(f"⚠️ Erro ao carregar o modelo: {e}. Usando modelo não treinado.")
    else:
        print(f"Aviso: Arquivo de modelo '{model_path}' não encontrado. Usando modelo não treinado.")
    model.to(device)
    model.eval()


# --- Login Manager ---
login_manager = LoginManager()
login_manager.login_view = 'auth.login'  # Aponta para o blueprint 'auth'
login_manager.login_message_category = 'info'


@login_manager.user_loader
def load_user(user_id):
    """Carrega usuário a partir do ID usando SQLite."""
    # Import aqui para evitar importação circular
    from .db_helpers import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()
    # Em SQLite usamos '?' como placeholder
    cursor.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()
    cursor.close()
    conn.close()
    if user_data:
        # graças ao row_factory=sqlite3.Row, podemos acessar por nome
        return User(user_data['id'], user_data['username'])
    return None


def create_app():
    # Ajusta os caminhos de static/templates para apontar para dentro de smt_app
    app = Flask(
        __name__,
        instance_relative_config=True,
        static_folder=os.path.join(os.path.dirname(__file__), 'static'),
        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    )

    # Garante que a pasta instance exista
    os.makedirs(app.instance_path, exist_ok=True)

    # Configurações básicas
    app.config.from_mapping(
        SECRET_KEY='uma_chave_se_muito_segura',  # Mova para instance/config.py em produção
        # Caminho do banco SQLite (novo)
        SQLITE_DB_PATH=os.path.join(app.instance_path, 'smt_inspection_new.db'),
        UPLOAD_FOLDER=os.path.join(app.static_folder, 'uploads'),
        DEBUG_FOLDER=os.path.join(app.static_folder, 'debug'),
        IMAGE_STORAGE=os.path.join(app.static_folder, 'images'),
        IMAGE_FOLDERS={
            'packages': os.path.join(app.static_folder, 'images', 'packages'),
            'results': os.path.join(app.static_folder, 'images', 'results'),
            'training': os.path.join(app.static_folder, 'images', 'training'),
            'uploads': os.path.join(app.static_folder, 'images', 'uploads'),
        },
    )

    # Garante que os diretórios existam
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DEBUG_FOLDER'], exist_ok=True)
    for folder in app.config['IMAGE_FOLDERS'].values():
        os.makedirs(folder, exist_ok=True)

    # Inicializa o LoginManager
    login_manager.init_app(app)

    # Carrega o modelo de IA
    load_trained_model()

    # Registra os Blueprints
    from . import routes_main
    from . import routes_auth
    from . import routes_product
    from . import routes_inspect

    app.register_blueprint(routes_main.bp)
    app.register_blueprint(routes_auth.bp)
    app.register_blueprint(routes_product.bp)
    app.register_blueprint(routes_inspect.bp)

    # Passa o modelo carregado para o app context para que os blueprints o vejam
    app.config['MODEL'] = model
    app.config['DEVICE'] = device

    return app
