from flask import Flask, request
from service.gen_answer import get_answer_from_ds
import json
app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/get_answer')
def get_answer():  # put application's code here
    query = request.args.get('query')
    response=get_answer_from_ds(query)
    return response

if __name__ == '__main__':
    app.run()
