import os
from flask import Flask, request, redirect, url_for, send_from_directory, jsonify, render_template, flash, Markup
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import re
import pickle
import pandas_profiling

UPLOAD_FOLDER = 'files/uploads'
ALLOWED_EXTENSIONS = set(['csv','xls','xlsx'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return ' '.join([m.group(0) for m in matches])

def is_binary(column):
    column = set(column)
    if len(column) > 2:
        return False
    else:
        return True

def is_numeric(column):
    numerics = (int, float, long)
    if all(isinstance(x, numerics) for x in column):
        return True
    else:
        return False

def uniqueness(column):
    ratio = len(set(column)) / float(len(column))
    if ratio >= 0.9:
        return 'too high'
    elif ratio < 0.9 and ratio >= 0.4:
        return 'high'
    elif ratio < 0.4 and ratio >=0.1:
        return 'medium'
    else:
        return 'small'

#TODO
def is_continuous_values(column):
    pass

def clean_unique(df):
    for column in df.columns:
        values = list(df[column].values)
        if not is_numeric(values):
            if uniqueness(values) in ['too high', 'high', 'medium']:
                print "Deleting non unique, non numeric", column
                del df[column]
        else:
            if uniqueness(values) in ['too high', 'high']:
                print "Deleting non unique, numeric", column
                del df[column]
    return df

def create_dummy_values(df):
    for column in df.columns:
        values = list(df[column].values)
        if not is_numeric(values):
            if is_binary(values):
                df[column] = df[column].apply(lambda x: 0 if x == values[0] else 1)
            else:
                df_temp = pd.get_dummies(df[column], prefix=column)
                df_temp.drop(df_temp.columns[len(df_temp.columns)-1], axis=1, inplace=True)
                df = df.join(df_temp)
                del df[column]
    return df
        # TODO
        # elif is_continuous_values(values[1]):


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
    filepath = app.config['UPLOAD_FOLDER']+'/'+filename
    if filepath.endswith('csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('xls','xlsx')):
        df = pd.read_excel(filepath)
    df.columns = [c.replace(' ', '_') for c in df.columns]
    if request.method == 'POST':
        column = request.form.get('option')
        pickle.dump(df, open('files/pkl/df.pkl', 'wb'))
        return redirect(url_for('model', filename=filename, column=column))

    return render_template('index.html', server_list=list(df.columns))
    # 

def get_numeric_columns(df):
    numerics = ['int', 'float']
    return df.select_dtypes(include=numerics).columns

@app.route('/uploads/<filename>/<column>/predict', methods=['GET', 'POST'])
def model(filename, column):
    if os.path.isfile('files/pkl/PREDICTOR-'+column+'.pkl') and os.path.isfile('files/pkl/data-'+column+'.pkl'):
        data = pickle.load(open('files/pkl/data-'+column+'.pkl', 'rb'))
    else:
        df = pickle.load(open('files/pkl/df.pkl', 'rb'))
        df.dropna(inplace=True)
        y = df[column]
        del df[column]
        df = clean_unique(df)

        include = list(df.columns)
        X = df[include]

        regressor_limits = []
        for regressor in X:
            regressor_limits.append({"name":regressor, "min": min(df[regressor]), "max": max(df[regressor]), "numeric": is_numeric(df[regressor].values), "values": df[regressor].unique()})

        df = create_dummy_values(df)
        include = list(df.columns)
        X = df[include]

        regressor_limits_prediction = include

        nc_columns = get_numeric_columns(df)
        nc = [x for x in nc_columns if x != column]
        df.ix[:, nc] = (df.ix[:, nc] - df.ix[:, nc].mean())\
                       / df.ix[:, nc].std()

        X = df[include]
        # y = df[column]

        values = list(y.values)

        if not is_numeric(values):
            PREDICTOR = RandomForestClassifier().fit(X, y)
        else:
            if uniqueness(values) == 'small':
                PREDICTOR = RandomForestClassifier().fit(X, y)
            else:
                PREDICTOR = RandomForestRegressor().fit(X, y)   

        data = {}
        data['predictor'] = camel_case_split(str(type(PREDICTOR)).split('.')[-1])
        data['score'] = PREDICTOR.score(X,y)
        data['stats'] = {'length':len(X), 'ratio':len(y[y==1])/float(len(y)), 'amount':len(y[y==1])}
        data['regressors'] = regressor_limits
        data['regressors_predict'] = regressor_limits_prediction
        data['prediction'] = column
        pickle.dump(PREDICTOR, open('files/pkl/PREDICTOR-'+column+'.pkl', 'wb'))
        pickle.dump(data, open('files/pkl/data-'+column+'.pkl', 'wb'))
    return render_template('result.html', data=data)
# @app.route('/uploads/<filename>')

@app.route('/uploads/<filename>/<column>/explore', methods=['GET', 'POST'])
def explore(filename, column):
    df = pickle.load(open('files/pkl/df.pkl', 'rb'))
    report = pandas_profiling.ProfileReport(df)
    out = Markup(report.html)
    return render_template('explore.html', report=out)

def normalize(val, mean, std):
    return (val - mean) / std

def output(data, column):
    PREDICTOR = pickle.load(open('files/pkl/PREDICTOR-'+column+'.pkl', 'rb'))
    render = []
    for x in data.values():
        render.append(x)
    # Change to floats where possible
    for i, x in enumerate(render):
        try:
            render[i] = float(x)
        except ValueError:
            pass
    # render = map(normalize, render, 4)

    # return render
    try:
        score = PREDICTOR.predict_proba(np.array(render).reshape(1, -1))
    except:
        score = PREDICTOR.predict(np.array(render).reshape(1, -1))        
    return str(score[0,1]*100)
    if score > 1 :
        score = 1
    elif score < 0:
        score = 0
    return round(score*100, 2)

def dummify(undummified, dummy_template):
    b = {x:0 for x in dummy_template}
    for key, value in undummified.iteritems():
        if key != 'column':
            new_key = key+'_'+value
            if new_key in b:
                b[new_key] = 1
            else:
                b[key] = value
    return b

#-------- ROUTES GO HERE -----------#
@app.route('/prediction')
def prediction():
    undummified = request.args
    column = undummified['column']
    dummified = pickle.load(open('files/pkl/data-'+column+'.pkl', 'rb'))
    dummified = dummified['regressors_predict']
    data = dummify(undummified, dummified)
    return jsonify(output(data, column))

@app.route('/json/<path:path>')
def send_json(path):
    return send_from_directory('json', path)


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.run(debug=True)
