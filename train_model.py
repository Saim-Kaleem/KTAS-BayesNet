from model import model
from utils import discretize_variables, save_cpds
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

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

    # Identify unique `Complain index` classes
    unique_classes = data['Complain index'].value_counts()
    unique_classes = unique_classes[unique_classes == 1].index

    # Separate unique classes and the rest of the data
    unique_data = data[data['Complain index'].isin(unique_classes)]
    rest_data = data[~data['Complain index'].isin(unique_classes)]

    print(f"Unique classes moved to training set: {len(unique_data)} samples")

    # Perform stratified split on the rest of the data
    train_data, test_data = train_test_split(
        rest_data, 
        test_size=0.2, 
        stratify=rest_data['Complain index'], 
        random_state=42
    )
    
    # Add unique examples to the training set
    train_data = pd.concat([train_data, unique_data], ignore_index=True)

    print(f"Final Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    # Check the minimum class size in the training data
    min_class_size = train_data['Complain index'].value_counts().min()
    if min_class_size < 6:
        print(f"Warning: Some classes have fewer than 2 samples. SMOTE will be skipped for these classes.")
        smote = SMOTE(k_neighbors=1, random_state=42)
    else:
        smote = SMOTE(k_neighbors=min(6, min_class_size - 1), random_state=42)
    
    # Apply SMOTE
    print("\nApplying SMOTE to balance the training data...")
    X_train = train_data.drop(columns=['Complain index'])
    y_train = train_data['Complain index']
    try:
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    except ValueError as e:
        print(f"SMOTE failed: {e}. Skipping SMOTE.")
        X_resampled, y_resampled = X_train, y_train
    
    # Combine the resampled features and target
    train_data_balanced = pd.concat([X_resampled, y_resampled], axis=1)
    
    # Discretize the training dataset
    print("\nPreprocessing data...")
    train_data_discrete = discretize_variables(train_data_balanced)

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