from model import model
from utils import discretize_variables, load_cpds
from pgmpy.inference import VariableElimination
import pandas as pd
from sklearn.model_selection import train_test_split

def predict_ktas(model, patient_data):
    """
    Predict KTAS level for a given patient
    
    Parameters:
    model: Trained BayesianNetwork object
    patient_data (dict): Dictionary containing patient information
    
    Returns:
    predicted_ktas: Most likely KTAS level
    probabilities: Probability distribution over KTAS levels
    """
    # Create inference object
    inference = VariableElimination(model)
    
    # Predict KTAS
    result = inference.query(variables=['KTAS_expert'], 
                           evidence=patient_data)
    
    return result.values, result

def evaluate_model(model, test_data):
    """
    Evaluate the model's performance on test data
    
    Parameters:
    model: Trained BayesianNetwork object
    test_data (pandas.DataFrame): Test dataset
    
    Returns:
    accuracy: Overall prediction accuracy
    """
    inference = VariableElimination(model)
    correct = 0
    total = len(test_data)
    
    for idx, row in test_data.iterrows():
        try:
            # Prepare evidence dictionary and convert values to match model states
            evidence = row.drop('KTAS_expert').to_dict()
            evidence = {key: int(value) if isinstance(value, float) else value for key, value in evidence.items()}
            
            # Get prediction
            result = inference.query(variables=['KTAS_expert'], evidence=evidence)
            
            # Access probabilities from the result
            probabilities = result.values  # NumPy array with probabilities
            
            # Find the predicted class (index + 1 corresponds to KTAS_expert levels)
            predicted_ktas = probabilities.argmax() + 1
            
            # Check if the prediction matches the actual value
            if predicted_ktas == row['KTAS_expert']:
                correct += 1
        
        except KeyError as e:
            print(f"KeyError at iteration {idx}: {e}")
            print(f"Evidence: {evidence}")
            continue  # Skip the problematic row and continue
    
    return correct / total

def main():
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('data_cleaned3.csv')
    
    # Select relevant columns
    data = df[['Sex', 'Age', 'Arrival mode', 'Injury', 'Complain index', 'Mental', 
               'Pain', 'NRS_pain', 'BP', 'HR', 'RR', 'BT', 'Saturation', 
               'KTAS_expert']]
    
    print(f"Dataset loaded with {len(data)} samples")
    
    # Print initial data info
    print("\nInitial data types:")
    print(data.dtypes)
    
    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Discretize both datasets
    print("\nPreprocessing data...")
    train_data_discrete = discretize_variables(train_data)
    test_data_discrete = discretize_variables(test_data)
    
    print("\nDiscretized data types:")
    print(train_data_discrete.dtypes)
    
    # Test the model
    print("\nTesting Bayesian Network...")
    try:
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

        # Evaluate the model on training data
        print("\nEvaluating model performance on training data...")
        accuracy = evaluate_model(trained_model, train_data_discrete)
        print(f"Model accuracy on training set: {accuracy:.2%}")
        
        # Evaluate the model on test data
        print("\nEvaluating model performance...")
        accuracy = evaluate_model(trained_model, test_data_discrete)
        print(f"Model accuracy on test set: {accuracy:.2%}")
        
        # Example prediction
        print("\nMaking sample prediction...")
        sample_patient = test_data_discrete.iloc[0].drop('KTAS_expert').to_dict()
        pred_probs, pred_dist = predict_ktas(trained_model, sample_patient)
        print("Prediction probabilities:", pred_probs)
        print("Actual KTAS:", test_data_discrete.iloc[0]['KTAS_expert'])
        
    except Exception as e:
        print(f"Error during model training/evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()