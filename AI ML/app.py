"""
E-commerce Customer Segmentation - Flask API + static site server
Project name: "E-commerce Customer Segmentation"
Run:  python app.py     ->  http://127.0.0.1:5000
"""
import os
import threading
import hashlib
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
import ml_pipeline as ml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_DATASET = os.path.join(DATA_DIR, "samples", "E-commerce_Customer_Segmentation_2026.csv")
MAX_UPLOAD_MB = 100

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

@app.after_request
def no_cache(resp):
    """Prototype: always serve fresh assets & API payloads."""
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp

STATE = {
    "dataset_path": DEFAULT_DATASET,
    "df": None,
    "load_error": None,
    "eda_cache": None,
    "eda_sig": None,
    "ml_cache": None,
    "ml_sig": None,
    "train_lock": threading.Lock(),
}


def load_dataset(path):
    df = pd.read_csv(path)
    return df


def get_df():
    """Load the dataset lazily; only reload if the file changed on disk."""
    global STATE
    path = STATE["dataset_path"]
    try:
        with open(path, "rb") as f:
            data_bytes = f.read()
    except FileNotFoundError:
        STATE["df"], STATE["load_error"] = None, f"Dataset file not found: {os.path.basename(path)}"
        return None
    df = pd.read_csv(path)
    STATE["df"], STATE["load_error"] = df, None
    return df


def _sig(df, path, mtime):
    h = hashlib.md5()
    h.update(f"{path}|{mtime}|{df.shape}".encode())
    return h.hexdigest()


def _file_sig(path):
    try:
        st = os.stat(path)
        h = hashlib.md5()
        h.update(f"{path}|{st.st_size}|{int(st.st_mtime)}".encode())
        return h.hexdigest()
    except FileNotFoundError:
        return None


@app.errorhandler(413)
def too_big(e):
    return jsonify({"error": f"File exceeds {MAX_UPLOAD_MB} MB limit"}), 413


@app.errorhandler(Exception)
def http_error(e):
    code = getattr(e, "code", 500)
    if isinstance(code, int) and code >= 400 and code < 500:
        return jsonify({"error": str(e)}), code
    import traceback
    traceback.print_exc()
    return jsonify({"error": f"Internal error: {type(e).__name__}: {e}"}), 500


# ------------------------------------------------------------------ static ---------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ------------------------------------------------------------------ API ------------------
@app.route("/api/health", methods=["GET"])
def health():
    path = STATE["dataset_path"]
    exists = os.path.exists(path)
    return jsonify({
        "status": "ok",
        "dataset_loaded": bool(exists),
        "dataset_name": os.path.basename(path),
        "dataset_is_default": path == DEFAULT_DATASET,
        "error": STATE["load_error"],
    })


@app.route("/api/dataset", methods=["GET"])
def api_dataset():
    df = get_df()
    if df is None:
        return jsonify({"error": STATE["load_error"]}), 404
    return jsonify({"name": os.path.basename(STATE["dataset_path"]), "info": ml.dataset_info(df)})


@app.route("/api/eda", methods=["GET"])
def api_eda():
    df = get_df()
    if df is None:
        return jsonify({"error": STATE["load_error"]}), 404
    sig = _file_sig(STATE["dataset_path"])
    if STATE["eda_cache"] is None or sig != STATE["eda_sig"]:
        STATE["eda_cache"] = ml.run_eda(df)
        STATE["eda_sig"] = sig
    return jsonify({"name": os.path.basename(STATE["dataset_path"]), "eda": STATE["eda_cache"]})


@app.route("/api/train", methods=["POST"])
def api_train():
    df = get_df()
    if df is None:
        return jsonify({"error": STATE["load_error"]}), 404
    if not STATE["train_lock"].acquire(blocking=False):
        return jsonify({"error": "A training run is already in progress. Please wait."}), 429
    try:
        sig = _file_sig(STATE["dataset_path"])
        if STATE["ml_cache"] is None or sig != STATE["ml_sig"]:
            STATE["ml_cache"] = ml.run_ml_pipeline(df)
            STATE["ml_sig"] = sig
        return jsonify(STATE["ml_cache"])
    finally:
        STATE["train_lock"].release()


@app.route("/api/upload", methods=["POST"])
def api_upload():
    global STATE
    if "file" not in request.files:
        return jsonify({"error": "No CSV file received. Field name must be 'file'."}), 400
    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv files are accepted."}), 400
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, "uploads", f.filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    f.save(dest)
    try:
        df = pd.read_csv(dest)
        if df.empty:
            raise ValueError("Uploaded file has zero rows.")
        needed = [c for c in (ml.CLUSTER_FEATURES + [ml.CLASS_TARGET])
                  if c not in df.columns]
        if needed:
            os.remove(dest)
            return jsonify({"error": f"Missing required column(s): {', '.join(needed)}"}), 400
    except Exception as exc:
        try: os.remove(dest)
        except OSError: pass
        return jsonify({"error": f"Could not parse CSV: {exc}"}), 400
    STATE["dataset_path"] = dest
    STATE["df"] = df
    STATE["eda_cache"], STATE["ml_cache"], STATE["load_error"] = None, None, None
    return jsonify({"status": "ok", "file": f.filename,
                    "rows": df.shape[0], "columns": df.shape[1],
                    "required_columns_ok": True})


if __name__ == "__main__":
    print("Starting E-commerce Customer Segmentation  ->  http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
