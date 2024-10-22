import json

from LanguageTranslator import LanguageTranslator

model = LanguageTranslator()

def lambda_handler(event, context):
    x = json.loads(event['body'])['englishSentence']

    # Make prediction
    y = model(x)

    return {
        'prediction': y,
    }
    