from flask import Flask, render_template, request
import pickle
import numpy as np
import cv2 as cv

#Função que carrega o modelo
with open('/home/frederico/Downloads/modelo_flores.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def mostrar_pred_flor_estoque():
    return render_template('predictorform.html')
@app.route('/previsao', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
       N_flor = request.files['files']
       N_flor_bytes = np.fromfile(N_flor, np.uint8)
       N_flor = cv.imdecode(N_flor_bytes, cv.IMREAD_COLOR)
       Nova_flor = []
       Nova_flor.append(np.array(cv.resize(N_flor, (224,224), interpolation = cv.INTER_AREA)))
       Nova_flor = np.array(Nova_flor)
       Nova_flor = Nova_flor/255
       pred = model.predict(Nova_flor)
       pred = np.argmax(pred, axis = 1)
       if (pred == 0):
         pred = 'Margarida'
       elif (pred == 1):
         pred = 'Dente-de-leão'
       elif (pred == 2):
         pred = 'Rosa'
       elif (pred == 3):
         pred = 'Girassol'
       elif (pred == 4):
         pred = 'Tulipa'
       return render_template('resultsform.html', Nova_flor=Nova_flor, pred_flor=pred)

@app.route('/sobre')
def sobre():
    return render_template('sobre.html')
