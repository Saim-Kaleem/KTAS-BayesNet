from model import model
from utils import discretize_variables, save_cpds
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd
from sklearn.model_selection import train_test_split

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
    train_data, test_data = train_test_split(data, test_size=0.2)
    
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
        
        # Save CPDs to a file
        save_cpds(trained_model, "cpds.json")

        # Check if model is valid
        if trained_model.check_model():
            print("Model validation passed")
        
        '''for cpd in model.get_cpds():
            if cpd.variable not in ['KTAS_expert']:
                print(f"\nCPD for {cpd.variable}:")
                print(cpd)'''
        
    except Exception as e:
        print(f"Error during model training/evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()