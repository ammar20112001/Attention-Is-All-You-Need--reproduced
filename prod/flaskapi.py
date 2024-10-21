from flask import Flask, request, render_template

from prod.LanguageTranslator import LanguageTranslator


model = LanguageTranslator()

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        x = request.form.get("englishSentence")
        y = model.predict(x)

        return render_template("page.html", english=x, french=y)

    return render_template("page.html")
