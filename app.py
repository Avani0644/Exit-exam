from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and feature information
bundle = joblib.load("shopping_model_bundle.joblib")
model = bundle["model"]
feature_columns = bundle["feature_columns"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]
categorical_options = bundle["categorical_options"]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            row = {}

            for column in feature_columns:
                value = request.form.get(column, "").strip()

                if value == "":
                    raise ValueError(f"Please enter a value for {column}.")

                if column in numeric_features:
                    row[column] = float(value)
                else:
                    # Match the HTML string back to the original category
                    matching_option = next(
                        (
                            option
                            for option in categorical_options[column]
                            if str(option) == value
                        ),
                        None
                    )

                    if matching_option is None:
                        raise ValueError(
                            f"Invalid selection for {column}."
                        )

                    row[column] = matching_option

            input_data = pd.DataFrame([row], columns=feature_columns)
            prediction = model.predict(input_data)[0]

        except (ValueError, TypeError) as exc:
            error = str(exc)

    return render_template(
        "predict.html",
        prediction=prediction,
        error=error,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        categorical_options=categorical_options
    )


if __name__ == "__main__":
    app.run(debug=True)