from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.after_request
def allow_private_network(response):
    # Lets a page hosted on GitHub Pages call this backend running on localhost.
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


def calculate(a, b, operation):
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return a / b
    raise ValueError("Unknown operation.")


@app.get("/")
def home():
    return jsonify({"status": "Flask backend is running"})


@app.post("/calculate")
def calculate_route():
    data = request.get_json(silent=True) or {}

    try:
        a = float(data.get("a"))
        b = float(data.get("b"))
    except (TypeError, ValueError):
        return jsonify({"error": "a and b must be numbers."}), 400

    try:
        result = calculate(a, b, data.get("operation"))
    except ZeroDivisionError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
