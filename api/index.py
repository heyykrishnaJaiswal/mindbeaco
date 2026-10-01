"""MindBeacon API (Flask). Pure Python: no numpy/sklearn needed at runtime."""
import json, math, os
from flask import Flask, request, jsonify, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "model.json")) as fh:
    M = json.load(fh)

app = Flask(__name__)

GROUP = {"age": "Age", "cgpa": "CGPA", "press": "High academic pressure", "sat": "Low study satisfaction",
         "fin": "Financial stress", "hrs": "Long study hours", "sleep_0": "Short sleep", "sleep_1": "Short sleep",
         "sleep_2": "Short sleep", "diet_0": "Eating habits", "diet_1": "Eating habits",
         "fam": "Family history", "sui": "Thoughts of self-harm"}
RANGES = {"age": (15, 60), "cgpa": (0, 10), "press": (0, 5), "sat": (0, 5), "fin": (1, 5), "hrs": (0, 12),
          "sleep": (0, 3), "diet": (0, 2), "fam": (0, 1), "sui": (0, 1)}


def clean(data):
    if not isinstance(data, dict):
        raise ValueError("Send a JSON object.")
    out = {}
    for k, (lo, hi) in RANGES.items():
        if k not in data:
            raise ValueError(f"Missing field: {k}")
        try:
            v = float(data[k])
        except (TypeError, ValueError):
            raise ValueError(f"Field {k} must be a number.")
        if not (lo <= v <= hi):
            raise ValueError(f"Field {k} must be between {lo} and {hi}.")
        out[k] = v
    return out


def predict(d):
    x = {k: d[k] for k in ("age", "cgpa", "press", "sat", "fin", "hrs", "fam", "sui")}
    for i in range(3):
        x[f"sleep_{i}"] = 1.0 if int(d["sleep"]) == i else 0.0
    for i in range(2):
        x[f"diet_{i}"] = 1.0 if int(d["diet"]) == i else 0.0
    z, groups = M["bias"], {}
    for i, c in enumerate(M["cols"]):
        v = M["coef"][i] * (x[c] - M["mean"][i]) / M["scale"][i]
        z += v
        groups[GROUP[c]] = groups.get(GROUP[c], 0.0) + v
    p = 1 / (1 + math.exp(-z))
    factors = sorted(((k, v) for k, v in groups.items() if v > 0.15), key=lambda e: -e[1])[:3]
    level = "high" if p >= 0.65 or d["sui"] == 1 else "moderate" if p >= 0.35 else "low"
    return {"p": round(p, 4), "level": level, "all": {k: round(v, 3) for k, v in groups.items()}, "factors": [[k, round(v, 3)] for k, v in factors]}


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        return jsonify(predict(clean(request.get_json(silent=True))))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/model-info")
def info():
    return jsonify({"rows": M["rows"], "metrics": M["metrics"]})


# Local development only: serve the frontend. On Vercel, /public is served by the CDN.
@app.route("/")
def home():
    return send_from_directory(os.path.join(ROOT, "public"), "index.html")


@app.route("/<path:name>")
def assets(name):
    return send_from_directory(os.path.join(ROOT, "public"), name)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
