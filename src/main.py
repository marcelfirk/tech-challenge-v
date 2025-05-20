import os
import sys
import pandas as pd
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from flask import Flask, send_from_directory, jsonify
from src.routes.prediction import create_prediction_route

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

PARQUET_PATH = "./src/data/applicants_processed.parquet"

def load_parquet(file_path):
    """Carregar o DataFrame a partir do .parquet."""
    try:
        return pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo {file_path} não encontrado.")
        return None
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {e}")
        return None

# Carregar o parquet já processado
applicants = load_parquet(PARQUET_PATH)

# Registrar a rota de prediction
app = create_prediction_route(app)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return jsonify({"message": "API de recrutamento está no ar. Use o endpoint /predict para predições."}), 200


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False)

