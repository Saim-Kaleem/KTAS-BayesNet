from model import model
from utils import discretize_variables, load_cpds
from pgmpy.inference import VariableElimination
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support

def evaluate_model(model, test_data, plot_confusion=True):
    """
    Comprehensive evaluation of the Bayesian Network model
    
    Parameters:
    model: Trained BayesianNetwork object
    test_data: Test dataset
    plot_confusion: Whether to plot confusion matrix
    
    Returns:
    dict: Dictionary containing various evaluation metrics
    """
    from pgmpy.inference import VariableElimination
    inference = VariableElimination(model)
    
    # Lists to store actual and predicted values
    y_true = []
    y_pred = []
    probabilities = []
    skipped_count = 0
    
    # Get predictions for all test cases
    for idx, row in test_data.iterrows():
        try:
            # Make prediction first
            evidence = row.drop('KTAS_expert').to_dict()
            result = inference.query(variables=['KTAS_expert'], evidence=evidence)
            
            # If prediction succeeds, then add both actual and predicted values
            pred_class = np.argmax(result.values) + 1
            pred_prob = result.values[pred_class - 1]
            
            # Only append actual value if prediction was successful
            actual = row['KTAS_expert']
            
            y_true.append(actual)
            y_pred.append(str(pred_class))
            probabilities.append(pred_prob)
            
        except Exception as e:
            skipped_count += 1
            print(f"Error at index {idx}: {e}")
            print(f"Evidence: {evidence}")
            continue
    
    print(f"\nProcessed {len(y_true)} samples successfully")
    print(f"Skipped {skipped_count} problematic samples")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate precision, recall, and F1 score for each class
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    
    # Plot confusion matrix if requested
    if plot_confusion:
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    # Calculate confidence of predictions
    mean_confidence = np.mean(probabilities)
    confidence_std = np.std(probabilities)
    
    # Prepare detailed metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'class_metrics': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        },
        'prediction_confidence': {
            'mean': mean_confidence,
            'std': confidence_std
        },
        'samples_processed': len(y_true),
        'samples_skipped': skipped_count
    }
    
    # Print summary metrics
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nMean Prediction Confidence: {mean_confidence:.4f} (±{confidence_std:.4f})")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Calculate per-class accuracy
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    print("\nPer-class Accuracy:")
    for i, acc in enumerate(per_class_accuracy):
        print(f"Class {i}: {acc:.4f}")
    
    return metrics

def analyze_errors(model, test_data, metrics):
    """
    Analyze prediction errors in detail
    
    Parameters:
    model: Trained BayesianNetwork object
    test_data: Test dataset
    metrics: Metrics dictionary from evaluate_model
    """
    inference = VariableElimination(model)
    
    # Get misclassified cases
    errors = []
    skipped_count = 0
    
    for idx, row in test_data.iterrows():
        try:
            # Convert actual to int, as it might be string
            actual = int(row['KTAS_expert'])
            evidence = row.drop('KTAS_expert').to_dict()
            result = inference.query(variables=['KTAS_expert'], evidence=evidence)
            
            # Add 1 to pred_class to match KTAS levels (1-5)
            pred_class = np.argmax(result.values) + 1
            pred_prob = result.values[pred_class - 1]
            
            if actual != pred_class:
                errors.append({
                    'index': idx,
                    'actual': actual,  # This is now guaranteed to be int
                    'predicted': pred_class,  # This is already int
                    'confidence': pred_prob,
                    'features': evidence
                })
        except Exception as e:
            skipped_count += 1
            print(f"Error at index {idx}: {e}")
            continue
    
    # Print error analysis
    print("\nError Analysis:")
    print(f"Total errors: {len(errors)}")
    print(f"Skipped samples: {skipped_count}")
    
    if errors:  # Only proceed if we have errors to analyze
        # Analyze common error patterns
        error_patterns = {}
        for error in errors:
            pattern = (error['actual'], error['predicted'])
            if pattern not in error_patterns:
                error_patterns[pattern] = 0
            error_patterns[pattern] += 1
        
        print("\nCommon Error Patterns (actual -> predicted):")
        for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"KTAS {pattern[0]} -> KTAS {pattern[1]}: {count} cases")
            
        # Calculate error severity (difference between actual and predicted KTAS levels)
        error_distances = [abs(int(error['actual']) - int(error['predicted'])) for error in errors]
        avg_error_distance = sum(error_distances) / len(error_distances)
        print(f"\nAverage KTAS Level Difference in Errors: {avg_error_distance:.2f}")
        print(f"Maximum KTAS Level Difference: {max(error_distances)}")
        
        # Distribution of error magnitudes
        error_magnitude_dist = {}
        for dist in error_distances:
            if dist not in error_magnitude_dist:
                error_magnitude_dist[dist] = 0
            error_magnitude_dist[dist] += 1
            
        print("\nError Magnitude Distribution:")
        for magnitude, count in sorted(error_magnitude_dist.items()):
            print(f"Off by {magnitude} level(s): {count} cases ({count/len(error_distances)*100:.1f}%)")
    else:
        print("No errors found in the analyzed samples.")
    
    return errors

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
    
    pred_class = np.argmax(result.values) + 1
    return pred_class, result.values

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
        
        '''for cpd in model.get_cpds():
            print(f"\nCPD for {cpd.variable}:")
            print(cpd)'''

        # Evaluate the model on training data
        print("\nEvaluating model performance on training data...")
        metrics = evaluate_model(trained_model, train_data_discrete)
        
        # Evaluate the model on test data
        print("\nEvaluating model performance...")
        metrics = evaluate_model(trained_model, test_data_discrete)

        # Analyze errors
        error_analysis = analyze_errors(trained_model, test_data_discrete, metrics)
        
        # Example prediction
        print("\nMaking sample prediction...")
        sample_patient = test_data_discrete.iloc[0].drop('KTAS_expert').to_dict()
        print("Patient data:", sample_patient)
        pred_probs, pred_dist = predict_ktas(trained_model, sample_patient)
        print("Prediction probabilities:", pred_probs)
        print("Actual KTAS:", test_data_discrete.iloc[0]['KTAS_expert'])
        
    except Exception as e:
        print(f"Error during model training/evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()