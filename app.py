from flask import Flask,request,url_for,render_template
from src.pipelines.predict_pipeline import CustomData,PredictPipeline
from src.utils import load_object
import os

app = Flask(__name__)

@app.route('/',methods=['GET'])
def form():
    return render_template('index.html',result="")

@app.route('end-to-end-students-performance.onrender.com//predict',methods=['POST'])
def predict():
    gender = request.form.get('gender')
    race_ethnicity = request.form.get('ethnicity')
    parental_level_of_education = request.form.get('parental_level_of_education')
    lunch = request.form.get('lunch')
    test_preparation_course = request.form.get('test_preparation_course')
    reading_score = int(request.form.get('reading_score'))
    writing_score = int(request.form.get('writing_score'))
    cd = CustomData(gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score)
    df = cd.get_data_as_data_frame()
    pp = PredictPipeline()
    prediction = pp.predict(df)
    return render_template('index.html',result=prediction[0])

# if __name__ == '__main__':
#     app.run(host="0.0.0.0",port=5001)
if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5001))
    app.run(host=host, port=port)