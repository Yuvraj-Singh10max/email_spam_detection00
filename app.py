from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

#========================loading the save files==================================================
model = pickle.load(open('logistic_regression.pkl','rb'))
feature_extraction = pickle.load(open('feature_extraction.pkl','rb'))


def predict_mail(input_text):
    input_user_mail  = [input_text]
    input_data_features = feature_extraction.transform(input_user_mail)
    prediction = model.predict(input_data_features)
    return prediction


@app.route('/', methods=['GET', 'POST'])
def analyze_mail():
    if request.method == 'POST':
        mail = request.form.get('mail')
        predicted_mail = predict_mail(input_text=mail)
        return render_template('index.html', classify=predicted_mail)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)