from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        try:
            # Collect data from the form
            data = CustomData(
                Age=float(request.form.get('Age')),
                Weight_kg=float(request.form.get('Weight_kg')),
                Height_m=float(request.form.get('Height_m')),
                Max_BPM=int(request.form.get('Max_BPM')),
                Avg_BPM=int(request.form.get('Avg_BPM')),
                Resting_BPM=int(request.form.get('Resting_BPM')),
                Session_Duration_hours=float(request.form.get('Session_Duration_hours')),
                Calories_Burned=float(request.form.get('Calories_Burned')),
                Fat_Percentage=float(request.form.get('Fat_Percentage')),
                Water_Intake_liters=float(request.form.get('Water_Intake_liters')),
                Workout_Frequency_days_week=int(request.form.get('Workout_Frequency_days_week')),
                Experience_Level=int(request.form.get('Experience_Level')),
                Gender=request.form.get('Gender'),
                Workout_Type=request.form.get('Workout_Type')
            )

            # Convert the data to DataFrame format
            final_new_data = data.get_data_as_dataframe()

            # Initialize the prediction pipeline
            predict_pipeline = PredictPipeline()

            # Make the prediction
            pred = predict_pipeline.predict(final_new_data)

            # Process the prediction result
            predicted_bmi = pred[0]

            return render_template('results.html', final_result=predicted_bmi)
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
