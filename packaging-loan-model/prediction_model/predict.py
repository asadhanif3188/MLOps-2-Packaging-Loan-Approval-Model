import pandas as pd
import numpy as np
from prediction_model.config import config
from prediction_model.data_handling import load_pipeline, load_dataset

classification_pipeline = load_pipeline(config.MODEL_NAME)

# def generate_prediction(data_input):
#     data = pd.DataFrame(data_input)
#     pred = classification_pipeline.predict(data[config.FEATURES])
#     output = np.where(pred == 1, 'Y', 'N')
#     result = {'Predictions': output}
#     return result

def generate_prediction():
    test_data = load_dataset(config.TEST_FILE)
    pred = classification_pipeline.predict(test_data[config.FEATURES])
    output = np.where(pred == 1, 'Y', 'N')
    # result = {'Predictions': output}
    # return result
    return output


if __name__ == '__main__':
    generate_prediction()