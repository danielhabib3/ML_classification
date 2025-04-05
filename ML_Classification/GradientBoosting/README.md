  df_test = df_test.apply(pd.to_numeric, errors='ignore')
[BASELINE] Accuracy avec toutes les colonnes : 92.62%

Sans 'date': Accuracy = 82.12% (Δ = -10.5%)
Sans 'hour': Accuracy = 93.04% (Δ = 0.42%)
Sans 'bc_price': Accuracy = 86.59% (Δ = -6.03%)
Sans 'bc_demand': Accuracy = 93.04% (Δ = 0.42%)
Sans 'ab_price': Accuracy = 92.62% (Δ = 0.0%)
Sans 'ab_demand': Accuracy = 92.93% (Δ = 0.31%)
Sans 'transfer': Accuracy = 92.79% (Δ = 0.17%)

--- Résumé ---
date                           -> Accuracy: 82.12% | Δ = -10.5%
bc_price                       -> Accuracy: 86.59% | Δ = -6.03%
ab_price                       -> Accuracy: 92.62% | Δ = 0.0%
transfer                       -> Accuracy: 92.79% | Δ = 0.17%
ab_demand                      -> Accuracy: 92.93% | Δ = 0.31%
hour                           -> Accuracy: 93.04% | Δ = 0.42%
bc_demand                      -> Accuracy: 93.04% | Δ = 0.42%


[]
Combo supprimé : [] --> Accuracy = 92.62%
['transfer']
Combo supprimé : ['transfer'] --> Accuracy = 92.79%
['ab_demand']
Combo supprimé : ['ab_demand'] --> Accuracy = 92.93%
['hour']
Combo supprimé : ['hour'] --> Accuracy = 93.04%
['bc_demand']
Combo supprimé : ['bc_demand'] --> Accuracy = 93.04%
['transfer', 'ab_demand']
Combo supprimé : ['transfer', 'ab_demand'] --> Accuracy = 93.1%
['transfer', 'hour']
Combo supprimé : ['transfer', 'hour'] --> Accuracy = 92.9%
['transfer', 'bc_demand']
Combo supprimé : ['transfer', 'bc_demand'] --> Accuracy = 93.56%
['ab_demand', 'hour']
Combo supprimé : ['ab_demand', 'hour'] --> Accuracy = 93.38%
['ab_demand', 'bc_demand']
Combo supprimé : ['ab_demand', 'bc_demand'] --> Accuracy = 93.52%
['hour', 'bc_demand']
Combo supprimé : ['hour', 'bc_demand'] --> Accuracy = 93.1%
['transfer', 'ab_demand', 'hour']
Combo supprimé : ['transfer', 'ab_demand', 'hour'] --> Accuracy = 92.97%
['transfer', 'ab_demand', 'bc_demand']
Combo supprimé : ['transfer', 'ab_demand', 'bc_demand'] --> Accuracy = 93.1%
['transfer', 'hour', 'bc_demand']
Combo supprimé : ['transfer', 'hour', 'bc_demand'] --> Accuracy = 92.97%
['ab_demand', 'hour', 'bc_demand']
Combo supprimé : ['ab_demand', 'hour', 'bc_demand'] --> Accuracy = 93.38%
['transfer', 'ab_demand', 'hour', 'bc_demand']
Combo supprimé : ['transfer', 'ab_demand', 'hour', 'bc_demand'] --> Accuracy = 93.14%



AVEC 5000 le meilleur est ab_demand bc_demeande puis ab_demand, bc_demand, transfer