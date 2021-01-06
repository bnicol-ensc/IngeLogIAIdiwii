from flask import Flask
import spacy 

app = Flask(__name__)
@app.route('/')
def hello():
    return {'Model 1' : 'functional, write your sentence after : /api/ to get classification results'}


@app.route('/api/<sentence>')
def hello_world(sentence):
    docs = nlp.tokenizer(sentence)

    scores, _ = textcat.predict([docs])
    label = scores.argmax()
    textcat.labels[label]

    return textcat.labels[label]
    

if __name__ == "__main__":
    nlp = spacy.load("./models/model1")
    textcat = nlp.get_pipe('textcat')

    app.run(debug=True)