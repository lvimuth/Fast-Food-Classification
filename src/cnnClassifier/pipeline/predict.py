import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.keras"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        # if result[0] == 1:
        #     prediction = 'Healthy'
        #     return [{ "image" : prediction}]
        # else:
        #     prediction = 'Coccidiosis'
        #     return [{ "image" : prediction}]
        
        predictions_map = {0: 'Baked Potato',
                            1: 'Burger',
                            2: 'Crispy Chicken',
                            3: 'Donut',
                            4: 'Fries',
                            5: 'Hot Dog',
                            6: 'Pizza',
                            7: 'Sandwich',
                            8: 'Taco',
                            9: 'Taquito'}

        # Get the prediction based on the result
        prediction = predictions_map.get(result[0], 'Unknown')

        # Return the result
        return [{ "image" : prediction }]