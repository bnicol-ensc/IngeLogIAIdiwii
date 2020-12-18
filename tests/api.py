from flask import Flask
import spacy 

app = Flask(__name__)
@app.route('/')
def hello():
    return {'Yolo' : 'Swag'}


@app.route('/api/<sentence>')
def hello_world(sentence):
    # texts = ["Are you ready for the tea party????? It's gonna be wild"]
    docs = nlp.tokenizer(sentence)

    # Use textcat to get the scores for each doc
    scores, _ = textcat.predict([docs])
    label = scores.argmax()
    textcat.labels[label]

    return textcat.labels[label]
    

if __name__ == "__main__":
    nlp = spacy.load("./models/model1")
    textcat = nlp.get_pipe('textcat')

    app.run(debug=True)