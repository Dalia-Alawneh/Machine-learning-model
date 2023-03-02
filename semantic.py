import itertools
from nltk.corpus import stopwords
import nltk
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

class Semantic:
    def __init__(self, text):
        self.text = text
        print(text)
    def process_text(self):
        # Tokenize the text
        tokens = nltk.word_tokenize(self.text)
        # Remove stop words
        stop_words = set(stopwords.words("english"))
        return [
            token
            for token in tokens
            if token.lower() not in stop_words
            and token != '.'
            and token != 'bit'
            and token != 'little'
            and token != 'also'
            and token != 'from'
            and token != 'suffer'
        ]
    def get_combinations(self):
        filtered_words = self.process_text()
        number = 3 if len(filtered_words)>1 else 1
        combinations = itertools.combinations(filtered_words, 3)
        list_of_combinations= []
        # convert the combinations to sentences
        for combination in combinations:
            sentence = ' '.join(combination)
            list_of_combinations.append(sentence)
        return list_of_combinations
    def getSimilarity(self,extracted_list):  
        # create the transform
        vectorizer = TfidfVectorizer()
    # encode the sentences
        vectors = vectorizer.fit_transform(extracted_list)
        return cosine_similarity(vectors[0], vectors[1])
    def get_symptoms(self):
        extracted_symptoms = []
        list_of_combinations=self.get_combinations()
        for symptom in symptoms:
        # check if the symptom is mentioned in the user text
            norm_symptom = symptom.replace("_", " ")
            for combin in list_of_combinations:
                if (
                self.getSimilarity([combin, norm_symptom]) > 0.40
                and symptom not in extracted_symptoms
            ):
                    extracted_symptoms.append(symptom)
        return extracted_symptoms


pickle.dump(Semantic, open(f'models/sem', 'wb'))