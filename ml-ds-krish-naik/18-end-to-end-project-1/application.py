from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Configure the templates folder as 'template' to match the project directory
application = Flask(__name__, template_folder="template")
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        # GET request renders the forms
        return render_template("index.html")

    # POST request processes the form data and runs predictions
    try:
        # Construct custom data object from form parameters
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )

        # Convert to pandas DataFrame matching preprocessor features
        pred_df = data.get_data_as_data_frame()

        # Run predict pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Format predicted math score
        predicted_score = round(results[0], 2)

        return render_template("index.html", prediction_text=f"{predicted_score}")
    except Exception as e:
        return render_template(
            "index.html", prediction_text=f"Error occurred during prediction: {str(e)}"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
