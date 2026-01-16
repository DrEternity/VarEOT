import papermill as pm

parameters = [
    ("MAN", "WOMAN", 1.0),
    ("MAN", "WOMAN", 0.1),
    ("WOMAN", "MAN", 1.0),
    ("WOMAN", "MAN", 0.1),
    ("ADULT", "CHILDREN", 1.0),
    ("CHILDREN", "ADULT", 1.0),
    ("ADULT", "CHILDREN", 0.1),
    ("CHILDREN", "ADULT", 0.1),
]

for param in parameters:
    INPUT_DATA = param[0] # MAN, WOMAN, ADULT, CHILDREN
    TARGET_DATA = param[1] # MAN, WOMAN, ADULT, CHILDREN
    EPSILON = param[2]
    pm.execute_notebook(
       'EgEOT_alae.ipynb',
       f'EgEOT_alae_i_{INPUT_DATA}_t_{TARGET_DATA}_e_{EPSILON}.ipynb',
       parameters=dict(
           INPUT_DATA=INPUT_DATA, 
           TARGET_DATA=TARGET_DATA, 
           EPSILON=EPSILON)
    )
