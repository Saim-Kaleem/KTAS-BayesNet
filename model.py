from pgmpy.models import BayesianNetwork

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
    ('HR', 'KTAS_expert'),
    ('RR', 'KTAS_expert'),
    ('BT', 'KTAS_expert'),
    ('Saturation', 'KTAS_expert')]
)