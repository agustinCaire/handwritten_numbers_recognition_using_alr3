from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from json import dumps
import numpy as np

import apiProccessing as proccessing



app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
   return render_template("index.html")



@app.route("/query", methods=['POST'])
def query():

    (res,label,querySkel) = proccessing.processQuery(request.get_json()['image'].split(',')[1])

    return dumps({"images": res, "label": str(label), "query": querySkel})

@app.route("/save", methods=['POST'])
def save():

    querySkel = proccessing.save(request.get_json()['image'].split(',')[1], request.get_json()['label'])

    return dumps({"query": querySkel})


if __name__ == '__main__':
    app.run(debug = True)