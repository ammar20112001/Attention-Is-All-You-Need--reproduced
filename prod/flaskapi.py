from flask import Flask

app = Flask(__name__)


@app.route("/")
def translate():
    return "<p>Hello, World!</p>"
