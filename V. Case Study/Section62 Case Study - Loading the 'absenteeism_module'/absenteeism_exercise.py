from absenteeism_module import *

model = absenteeism_model('model', 'scaler')
# a file, containing a fine-tuned finalized version of the logistic regression model

model.load_and_clean_data('Absenteeism_new_data.csv')
# .load_and_clean_data() will preprocess the entire data set we provide

model.predicted_outputs()
# .predicted_outputs() - its roke is to feed the cleaned data into the model, and deliver the output we discussed

model.predicted_outputs().to_csv('Absenteeism_predictions.csv', index=False)
