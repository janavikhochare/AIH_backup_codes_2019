from flask import Flask, render_template, url_for, request, session, redirect
import warnings
import json
import wind
# import version_1

warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_data', methods=['POST'])
def handle_data():
    projectpath = request.form['projectFilepath']
    rmse, acc = wind.deploy(projectpath)
    return render_template('results.html', rmse=rmse, acc=acc)

@app.route('/frontend')
def home():
    lat, lon, y_pred, y_dir = wind.deploy('INM00042103')
    y_pred = round(y_pred, 2)
    print(lat, lon, y_pred, y_dir)
    variables = {
        "page_title": "Document",
        "page_desc": "Some Description",
        "data_list": json.dumps([
                                 {
                                 "latitude": lat/10000,
                                 "longitude": lon/10000,
                                 "popup_html": "<h5>Loaction1</h5>Wind speed: " + str(y_pred) +  " kmph<br>Wind direction: " + str(y_dir)
                                 },
                                 {
                                 "latitude": 303833.0,
                                 "longitude": 767667.0,
                                 "popup_html": "<h5>Loaction2</h5>Wind speed: 12.5 kmph<br>Wind direction: East"
                                 },
                                 {
                                 "latitude": -50009.51,
                                 "longitude": -177778.52,
                                 "popup_html": "<h5>Loaction3</h5>Wind speed: 12.5 kmph<br>Wind direction: East"
                                 },
                                 ])
    }
    return render_template('home.html', **variables)
    # df = version_1.runs()
    # lat = list(df['latitude'])[0]
    # lon = list(df['longitude'])[0]
    # y_pred = round(df.loc[0, [w]], 2)
    # print(lat, lon, y_pred, y_dir)
    # variables = {
    #     "page_title": "Document",
    #     "page_desc": "Some Description",
    #     "data_list": json.dumps([
    #                              {
    #                              "latitude": lat/10000,
    #                              "longitude": lon/10000,
    #                              "popup_html": "<h5>Loaction1</h5>Wind speed: " + str(y_pred) +  " kmph<br>Wind direction: " + str(y_dir)
    #                              }
    #                              ])
    # }
    # return render_template('home.html', **variables)

if __name__ == '__main__':
    app.secret_key = 'mysecret'
    app.debug = True
    app.run(host='localhost', port="5001")

