# Meilleurs résultats actuels

num_estimators : 200000

other_column_to_drop = ['hc', 'weight_min',  'ptcl',  'max_power', 'weight_max', 'co']

Public score: 0.0644

-> Sans 'brand', 'model', 'car_class', 'grbx_type_ratios'

-----------------------------------

num_estimators : 120000

other_column_to_drop = ['hc', 'weight_min',  'ptcl',  'max_power', 'weight_max', 'co']


Public score: 0.0646

-> Sans 'brand', 'model', 'car_class', 'grbx_type_ratios'


# TESTS A EFFECTUER


--> NE pas rajouter 'MODEL' mais avec grbx_type_ratios c'est proche



---------------------------------------------------
   n_estimators       MAE
0         50000  0.076891
1        100000  0.076902
2        200000  0.076908
---------------------------------------------------




TEST DIFFERENT n_estimators :
n_estimators: 10000 → MAE: 0.1007  
test 11000  
n_estimators: 11000 → MAE: 0.0991  
test 12000  
n_estimators: 12000 → MAE: 0.0978  
test 13000  
n_estimators: 13000 → MAE: 0.0967  
test 14000  
n_estimators: 14000 → MAE: 0.0957  
test 15000  
n_estimators: 15000 → MAE: 0.0949  
test 16000  
n_estimators: 16000 → MAE: 0.0942  
test 17000  
n_estimators: 17000 → MAE: 0.0934  
test 18000  
n_estimators: 18000 → MAE: 0.0930  
test 19000  
n_estimators: 19000 → MAE: 0.0926  

n_estimators: 20000 → MAE: 0.09206
test 21000
n_estimators: 21000 → MAE: 0.09180
test 22000
n_estimators: 22000 → MAE: 0.09133
test 23000

TEST feature :

A TESTER : ENLEVER WEIGHT_MIN, PTCL, hc


📊 Impact des colonnes (trié par impact absolu sur MAE):
overall_cons              → MAE: 0.1580 | Différence: +0.0396
urb_cons                  → MAE: 0.1543 | Différence: +0.0359
exturb_cons               → MAE: 0.1333 | Différence: +0.0148
ptcl                      → MAE: 0.1065 | Différence: -0.0119
fuel_type_ES/GN           → MAE: 0.1070 | Différence: -0.0115
weight_min                → MAE: 0.1070 | Différence: -0.0115
range_MOY-INFER           → MAE: 0.1082 | Différence: -0.0102
hc                        → MAE: 0.1084 | Différence: -0.0100
fuel_type_GH              → MAE: 0.1095 | Différence: -0.0090
range_LUXE                → MAE: 0.1103 | Différence: -0.0081
fuel_type_GN/ES           → MAE: 0.1263 | Différence: +0.0079
max_power                 → MAE: 0.1106 | Différence: -0.0078
range_MOY-INFERIEURE      → MAE: 0.1118 | Différence: -0.0067
weight_max                → MAE: 0.1125 | Différence: -0.0060
range_SUPERIEURE          → MAE: 0.1126 | Différence: -0.0058
range_MOY-SUPER           → MAE: 0.1127 | Différence: -0.0058
co                        → MAE: 0.1127 | Différence: -0.0058
fuel_type_GP/ES           → MAE: 0.1236 | Différence: +0.0052
hcnox                     → MAE: 0.1227 | Différence: +0.0042
fuel_type_GN              → MAE: 0.1148 | Différence: -0.0037
fuel_type_ES              → MAE: 0.1214 | Différence: +0.0030
range_INFERIEURE          → MAE: 0.1164 | Différence: -0.0020
fuel_type_EH              → MAE: 0.1171 | Différence: -0.0013
fuel_type_GO              → MAE: 0.1198 | Différence: +0.0013
hybrid_oui                → MAE: 0.1171 | Différence: -0.0013
fuel_type_FE              → MAE: 0.1195 | Différence: +0.0011
nox                       → MAE: 0.1176 | Différence: -0.0009
fuel_type_ES/GP           → MAE: 0.1176 | Différence: -0.0008
fuel_type_GL              → MAE: 0.1177 | Différence: -0.0008



Testing with columns removed: ['weight_min']
Gradient Boosting Regressor MAE: 0.0822

Testing with columns removed: ['ptcl']
Gradient Boosting Regressor MAE: 0.0800

Testing with columns removed: ['hc']
Gradient Boosting Regressor MAE: 0.0827


Testing with columns removed: ['ptcl', 'weight_min', 'hc']
Gradient Boosting Regressor MAE: 0.0734

Testing with columns removed: ['hc', 'weight_min']
Gradient Boosting Regressor MAE: 0.0789

Testing with columns removed: ['ptcl', 'weight_min']
Gradient Boosting Regressor MAE: 0.0759

Testing with columns removed: ['ptcl', 'hc']
Gradient Boosting Regressor MAE: 0.0843
