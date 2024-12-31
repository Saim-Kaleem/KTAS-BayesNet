from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Define the network structure
edges = [
    # Demographics affect presentation
    ('Sex', 'Pain'),
    ('Age', 'Pain'),
    ('Age', 'BP'),
    ('Age', 'HR'),
    
    # Injury and arrival relationships
    ('Injury', 'Pain'),
    ('Injury', 'NRS_pain'),
    ('Arrival mode', 'Injury'),
    
    # Vital signs relationships
    ('Pain', 'HR'),  # Pain affects heart rate
    ('Pain', 'RR'),  # Pain affects respiratory rate
    ('Mental', 'RR'), # Mental status affects breathing
    
    # Core physiological relationships
    ('HR', 'Saturation'),
    ('RR', 'Saturation'),
    ('BT', 'HR'),    # Temperature affects heart rate
    
    # Complaint relationships
    ('Pain', 'Complain index'),
    ('Mental', 'Complain index'),
    ('Injury', 'Complain index'),
    
    # KTAS determinants based on protocol
    ('Complain index', 'KTAS_expert'),
    ('Mental', 'KTAS_expert'),
    ('Pain', 'KTAS_expert'),
    ('NRS_pain', 'KTAS_expert'),
    ('BP', 'KTAS_expert'),
    ('HR', 'KTAS_expert'),
    ('RR', 'KTAS_expert'),
    ('BT', 'KTAS_expert'),
    ('Saturation', 'KTAS_expert')
]

# Create the model
model = BayesianNetwork(edges)

def train_model(data):
    """
    Train the Bayesian Network using Maximum Likelihood Estimation
    
    Parameters:
    data (pandas.DataFrame): Training data with all required columns
    
    Returns:
    trained_model: Trained BayesianNetwork object
    """
    try:
        # Estimate CPDs using MLE
        estimator = MaximumLikelihoodEstimator(model, data)
        
        # Get CPDs for all nodes
        for node in model.nodes():
            cpd = estimator.estimate_cpd(node)
            model.add_cpds(cpd)
        
        return model
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        print("Data types of columns:")
        for col in data.columns:
            print(f"{col}: {data[col].dtype}")
            print(f"Unique values: {data[col].unique()}")
        raise

def discretize_variables(data):
    """
    Convert all variables to string type for pgmpy compatibility
    since data is already preprocessed and categorized
    """
    # Create copy to avoid modifying original data
    df = data.copy()
    
    # Convert all columns to string for pgmpy compatibility
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    return df

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
    from pgmpy.inference import VariableElimination
    
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
    from pgmpy.inference import VariableElimination
    
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
    
    # Train the model
    print("\nTraining Bayesian Network...")
    try:
        trained_model = train_model(train_data_discrete)
        print("Model training completed successfully")
        
        # Check if model is valid
        if trained_model.check_model():
            print("Model validation passed")
            print(model.get_cpds('KTAS_expert').state_names)

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