from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import json
from flask import request
from model import model
from utils import discretize_variables, load_cpds
from evaluate_model import predict_ktas

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def send_static(path):
    print("Requested path:", path)
    links = ["","login","dashboard"]
    if path in links:
        return render_template('index.html')
    else:
        return send_from_directory('static', path)


@app.route('/api/get_bayes', methods=['POST'])
def get_bayes():
    try:
        data = request.get_json()

        # Load the model
        trained_model = model
        load_cpds(trained_model, "cpds.json")
        print("Model loaded successfully")
        
        # Check if model is valid
        if trained_model.check_model():
            print("Model validation passed")
        
        for cpd in model.get_cpds():
            print(f"\nCPD for {cpd.variable}:")
            print(cpd)
        
        # Get prediction
        ktas_prediction, probabilities = predict_ktas(
            model=trained_model,
            patient_data=data
        )
        
        # Convert numpy values to Python native types for JSON serialization
        response = {
            "ktas_prediction": int(ktas_prediction),
            "probabilities": probabilities.tolist()
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


"""uSHFUIH79Ahfu7sfvhudshauioshcvuasdhfvuifdahvauihfiPHAFCU89DHFCUIASDHCVUPHASUIDVHUASVHUIADFVHDFUIAHVADFUIHV
DSBSDVBYSDVYUASGYUGAYVGYDFVBADFVBHADFHVDFUIHVUIDFVHADFUIVHDFUAHVHUAHVUAHSDVNJIASDN"""

#webbrowser.open("http://localhost:" + str(PORT), new=0, autoraise=True)
PORT = 5000
app.run(port=PORT, debug=True)