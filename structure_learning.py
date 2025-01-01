import pandas as pd
from pgmpy.estimators import PC, HillClimbSearch, TreeSearch, ExhaustiveSearch, MmhcEstimator, ExpertInLoop

# Load dataset
data = pd.read_csv("data_cleaned3.csv")
data = data[['Sex', 'Age', 'Arrival mode', 'Injury', 'Complain index', 'Mental', 
               'Pain', 'NRS_pain', 'BP', 'HR', 'RR', 'BT', 'Saturation', 
               'KTAS_expert']]

# Display basic info about the dataset
print("Dataset Info:")
print(data.info())
print(data.head())

# list of structure learning instances
est = [PC(data), HillClimbSearch(data), TreeSearch(data), ExhaustiveSearch(data), ExpertInLoop(data), MmhcEstimator(data)]

# list of all scoring / constraint methods
scoring_methods = ['k2score', 'bdeuscore', 'bicscore', 'bdsscore', 'aicscore']

for idx, e in enumerate(est):
    print(idx)
    print(f"\nEstimator: {e}")
    for s in scoring_methods:
        try:
            if idx == 2:
                dag = e.estimate(estimator_type='chow-liu')
                print(dag.edges())
            elif idx == 3:
                dag = e.estimate()
                print(dag.edges())
            else:
                print(f"Scoring method: {s}")
                dag = e.estimate(scoring_method=s)
                print(dag.edges())
        except Exception as ex:
            print(f"Error: {ex}")
            continue

# Create a BayesianNetwork object from the learned structure
from pgmpy.models import BayesianNetwork
model = BayesianNetwork(dag)

# Optional: Validate the learned model
if model.check_model():
    print("The learned model is valid.")
else:
    print("The learned model is invalid. Check the dataset or constraints.")

# Optionally visualize the structure (requires `networkx` and `matplotlib`)
try:
    import networkx as nx
    import matplotlib.pyplot as plt

    nx.draw(model, with_labels=True, node_size=3000, node_color="lightblue", font_size=12, font_weight="bold")
    plt.title("Learned Bayesian Network Structure")
    plt.show()
except ImportError:
    print("Visualization skipped; install networkx and matplotlib for graph visualization.")