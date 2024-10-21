from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method=="POST":
        english_sentence = request.form.get("englishSentence")
        print(english_sentence)
        return render_template("page.html")

    return render_template('page.html')
