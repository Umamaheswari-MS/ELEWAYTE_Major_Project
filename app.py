from flask import Flask, render_template, request
import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the saved Random Forest model
file = open("D:/foruma/newmodel.pkl","rb")
rf_model = pickle.load(file)
file.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = []
        input_columns = [
            'General_Health', 'Checkup', 'Exercise', 'Skin_Cancer', 'Other_Cancer',
            'Depression', 'Diabetes', 'Arthritis', 'Sex', 'Age_Category',
            'Height_(cm)', 'Weight_(kg)', 'BMI', 'Smoking_History',
            'Alcohol_Consumption', 'Fruit_Consumption', 'Green_Vegetables_Consumption',
            'FriedPotato_Consumption'
        ]
        
        for column in input_columns:
            value = float(request.form[column])
            input_data.append(value)

        new_input = np.array(input_data).reshape(1, -1)
        prediction = rf_model.predict(new_input)

        # Convert prediction to meaningful result
        if prediction == 0:
            result = "No heart disease"
        elif prediction == 1:
            result = "Heart disease present"
        else:
            result = "Unknown prediction"

        return render_template('index.html', prediction_result=result)

    return render_template('index.html', prediction_result=None)

if __name__ == '__main__':
    app.run(debug=True)
