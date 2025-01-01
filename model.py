from pgmpy.models import BayesianNetwork
from daft import PGM

model = BayesianNetwork(
    [# Demographics affect presentation
    ('Sex', 'Pain'),
    ('Age', 'Pain'),
    ('Age', 'BP'),
    ('Age', 'HR'),
    ('Age', 'Injury'),
    
    # Injury and arrival relationships
    ('Injury', 'Pain'),
    ('Arrival mode', 'Injury'),
    ('Pain', 'NRS_pain'),  # Pain affects pain score
    ('Pain', 'Mental'),    # Pain affects mental status
    
    # Vital signs relationships
    ('Pain', 'HR'),  # Pain affects heart rate
    ('Pain', 'RR'),  # Pain affects breathing
    ('Pain', 'BP'),  # Pain affects blood pressure
    ('Mental', 'RR'), # Mental status affects breathing
    
    # Core physiological relationships
    ('HR', 'Saturation'),
    ('BT', 'HR'),    # Temperature affects heart rate
    
    # Complaint relationships
    ('NRS_pain', 'Complain index'),
    ('Mental', 'Complain index'),
    ('Injury', 'Complain index'),
    
    # KTAS determinants based on protocol
    ('Complain index', 'KTAS_expert'),
    ('NRS_pain', 'KTAS_expert'),
    ('BP', 'KTAS_expert'),
    ('Mental', 'KTAS_expert'),
    ('HR', 'KTAS_expert'),
    ('RR', 'KTAS_expert'),
    ('BT', 'KTAS_expert'),
    ('Saturation', 'KTAS_expert')]
)

def main():
    # Initialize the PGM with layout parameters
    pgm = PGM(aspect=1.5, node_unit=1.5)

    # Add nodes
    pgm.add_node("Sex", "Sex", 0, 4)
    pgm.add_node("Age", "Age", 1, 4)
    pgm.add_node("Arrival mode", "Arrival Mode", 2, 4)
    pgm.add_node("Injury", "Injury", 1, 3)
    pgm.add_node("Pain", "Pain", 1, 2)
    pgm.add_node("Mental", "Mental", 2, 2)
    pgm.add_node("NRS_pain", "NRS Pain", 0, 1)
    pgm.add_node("Complain index", "Complain Index", 2, 1)
    pgm.add_node("HR", "Heart Rate", 3, 3)
    pgm.add_node("RR", "Respiratory Rate", 3, 2)
    pgm.add_node("BP", "Blood Pressure", 2, 3)
    pgm.add_node("BT", "Body Temp", 4, 3)
    pgm.add_node("Saturation", "Saturation", 4, 2)
    pgm.add_node("KTAS_expert", "KTAS Expert", 3, 1)

    # Add edges
    pgm.add_edge("Sex", "Pain")
    pgm.add_edge("Age", "Pain")
    pgm.add_edge("Age", "BP")
    pgm.add_edge("Age", "HR")
    pgm.add_edge("Age", "Injury")
    pgm.add_edge("Injury", "Pain")
    pgm.add_edge("Arrival mode", "Injury")
    pgm.add_edge("Pain", "NRS_pain")
    pgm.add_edge("Pain", "Mental")
    pgm.add_edge("Pain", "HR")
    pgm.add_edge("Pain", "RR")
    pgm.add_edge("Pain", "BP")
    pgm.add_edge("Mental", "RR")
    pgm.add_edge("HR", "Saturation")
    pgm.add_edge("BT", "HR")
    pgm.add_edge("NRS_pain", "Complain index")
    pgm.add_edge("Mental", "Complain index")
    pgm.add_edge("Injury", "Complain index")
    pgm.add_edge("Complain index", "KTAS_expert")
    pgm.add_edge("NRS_pain", "KTAS_expert")
    pgm.add_edge("BP", "KTAS_expert")
    pgm.add_edge("Mental", "KTAS_expert")
    pgm.add_edge("HR", "KTAS_expert")
    pgm.add_edge("RR", "KTAS_expert")
    pgm.add_edge("BT", "KTAS_expert")
    pgm.add_edge("Saturation", "KTAS_expert")

    # Render and save the PGM
    pgm.render()
    pgm.figure.savefig("bayesian_network_daft.png")
    print("Graph saved as 'bayesian_network_daft.png'.")

if __name__ == '__main__':
    main()