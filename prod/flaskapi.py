import torch

from flask import Flask, request, render_template

from prod.LanguageTranslator import LanguageTranslator


model = LanguageTranslator()

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method=="POST":
        x = request.form.get("englishSentence")
        y = model.predict(x)
        print(y)

        return render_template("page.html")

    return render_template('page.html')
