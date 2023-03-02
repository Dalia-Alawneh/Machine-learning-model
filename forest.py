import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

# data fram for training
df = pd.read_csv('data/Training - Training.csv')

disease=['Brain and nerves clinic','Dermatology clinic','Ear, nose and throat clinic','Emergency','General clinic',
'Heart, veins and arteries clinic','Internal clinic','Oncology clinic ','Orthopedic clinic','Respiratory clinic','Urology clinic', 'Endocrinology clinic','Neurology and brain clinic']

# replace lable
# df.replace({'prognosis':{'Brain and nerves clinic':0,'Dermatology clinic':1,'Ear, nose and throat clinic':2,'Emergency':3,'General clinic':4,
# 'Heart, veins and arteries clinic':5,'Internal clinic':6,'Oncology clinic':7,'Orthopedic clinic':8,'Respiratory clinic':9,'Urology clinic':10, 'Endocrinology clinic':11,'Neurology and brain clinic':12}},inplace=True)
#split data
X = df.drop('prognosis', axis = 1)
y = df['prognosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 40)
np.ravel(y_train)


# define a list of symptoms
symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions',
            'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
            'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
            'vomiting', 'burning_micturition', 'spotting_ urination',
            'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets',
            'mood_swings', 'weight_loss', 'restlessness', 'lethargy',
            'patches_in_throat', 'irregular_sugar_level', 'cough',
            'high_fever', 'sunken_eyes', 'breathlessness', 'sweating',
            'dehydration', 'indigestion', 'headache', 'yellowish_skin',
            'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
            'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
            'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
            'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
            'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision',
            'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
            'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
            'fast_heart_rate', 'pain_during_bowel_movements',
            'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus',
            'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity',
            'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
            'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
            'excessive_hunger', 'extra_marital_contacts',
            'drying_and_tingling_lips', 'slurred_speech', 'knee_pain',
            'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
            'swelling_joints', 'movement_stiffness', 'spinning_movements',
            'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
            'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
            'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
            'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
            'altered_sensorium', 'red_spots_over_body', 'belly_pain',
            'abnormal_menstruation', 'dischromic _patches',
            'watering_from_eyes', 'increased_appetite', 'polyuria',
            'family_history', 'mucoid_sputum', 'rusty_sputum',
            'lack_of_concentration', 'visual_disturbances',
            'receiving_blood_transfusion', 'receiving_unsterile_injections',
            'coma', 'stomach_bleeding', 'distention_of_abdomen',
            'history_of_alcohol_consumption', 'fluid_overload.1',
            'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
            'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
            'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
            'inflammatory_nails', 'blister', 'red_sore_around_nose',
            'yellow_crust_ooze']

#####################################  new   ################
forest = RandomForestClassifier(random_state = 42, max_features = 'sqrt', n_jobs = 1, verbose = 1).fit(X_train, np.ravel(y_train))
# forest = AdaBoostClassifier(random_state = 42).fit(X_train, np.ravel(y_train))

# print('Training set score: {:.4f}'.format(forest.score(X_train, y_train)))
# print('Test set score: {:.4f}'.format(forest.score(X_test, y_test)))

y_pred=forest.predict(X_test)
forest_predict = forest.predict(X_test)

pickle.dump(forest, open(f'models/forest', 'wb'))
