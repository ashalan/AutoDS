import os
from flask import Flask, request, redirect, url_for, send_from_directory, jsonify, render_template
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import re
import pickle

UPLOAD_FOLDER = 'files/uploads'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_binary_values(column):
    column = set(column)
    if len(column) > 2:
        return (False, list(column))
    else:
        return (True, list(column))

#TODO
def is_continuous_values(column):
    print hello

def create_dummy_values(df):
    for column in df.columns:
        values = is_binary_values(df[column].values)
        if values[0] == True:
            df[column] = df[column].apply(lambda x: 0 if x == values[1][0] else 1)
        # TODO
        # else:
        #     if is_continuous_values(values[1]):

def output(data):
    PREDICTOR = pickle.load(open('files/pkl/PREDICTOR.pkl', 'rb'))
    render = []
    for x in data.values():
        render.append(x)
    render = map(float, render)
    # return render
    try:
        score = PREDICTOR.predict_proba(np.array(render).reshape(1, -1))
        results = {'survival chances': score[0,1], 'death chances': score[0,0]}
        print results
    except:
        score = PREDICTOR.predict(np.array(render).reshape(1, -1))        
    print score
    return str(score[0,1]*100)
#TODO
# def convert_to_numerics(df):
#     for column in df.columns:
#         values = column.unique()
#         try values = 

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('column', filename=filename))
    return render_template('home.html')

@app.route('/uploads/<filename>', methods=['GET', 'POST'])
def column(filename):
    df = pd.read_csv(app.config['UPLOAD_FOLDER']+'/'+filename)
    numerics = ['int', 'float']
    df = df.select_dtypes(include=numerics)

    if request.method == 'POST':
        column = request.form.get('option')
        return model(column, df)

    return render_template('index.html', server_list=list(df.columns))
    # 

def model(column, df):
    nc = [x for x in df.columns if x != column]
    df.ix[:, nc] = (df.ix[:, nc] - df.ix[:, nc].mean())\
                   / df.ix[:, nc].std()
    include = list(df.columns)
    include.remove(column)

    # Drop NaNs
    create_dummy_values(df)
    df = df.dropna()

    X = df[include]
    y = df[column]
    try:
        PREDICTOR = RandomForestClassifier().fit(X, y)
    except:
        PREDICTOR = RandomForestRegressor().fit(X, y)
    data = {}
    data['predictor'] = re.compile('[^a-zA-Z]').sub('', str(type(PREDICTOR)).split('.')[-1])
    data['score'] = (PREDICTOR.score(X,y))
    data['regressors'] = zip(include, PREDICTOR.feature_importances_)
    data['prediction'] = column
    pickle.dump(PREDICTOR, open('files/pkl/PREDICTOR.pkl', 'wb'))
    return render_template('result.html', data=data)
# @app.route('/uploads/<filename>')

#-------- ROUTES GO HERE -----------#
@app.route('/prediction')
def prediction():
    data = request.args
    return jsonify(output(data))

@app.route('/json/<path:path>')
def send_json(path):
    return send_from_directory('json', path)


if __name__ == '__main__':
    app.run(debug=True)
