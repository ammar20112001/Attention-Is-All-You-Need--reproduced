import json

import requests

from flask import Flask, request, render_template


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        x = request.form.get("englishSentence")

        # POST request to aws lambda
        lambda_url = (
            "https://f652c6u5ib2azi42ccfjwxf4ca0rfmwc.lambda-url.eu-north-1.on.aws/"
        )

        headers = {"Content-Type": "application/json"}
        payload = json.dumps({"englishSentence": x})

        response = requests.post(  # we POST the english sentence to the URL, expecting a prediction as a response
            lambda_url, data=payload, headers=headers
        )
        pred = response.json()  # the response is also json

        return render_template("page.html", english=x, french=pred["prediction"])

    return render_template("page.html")
