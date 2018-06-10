import cv2
import numpy as np
from PIL import Image
import time

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


# All our models are trained with image size 56x56.
target_size = (56, 56)
NUM_CLASSES = 14
pieceTypeList = ['b_jiang','b_ju', 'b_ma', 'b_pao', 'b_shi', 'b_xiang', 'b_zu',
                'r_bing', 'r_ju', 'r_ma', 'r_pao', 'r_shi', 'r_shuai', 'r_xiang']


def predict(model, img, target_size, top_n=3):
    """Run model prediction on image
    Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
    top_n: # of top predictions to return
    Returns:
    list of predicted labels and their probabilities
    """
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    preds = model.predict(x)
    pred_piece = np.where(preds[0] == 1)[0][0]

    print(preds)
    print(pieceTypeList[pred_piece])
    return pieceTypeList[pred_piece]


if __name__=="__main__":
    # model = loadCNN(), here we select the baseline model for evaluation
    model = load_model('./toy_cnn_mini_model_baseline.h5')

    piece_type_prediction = 'Not ready'
    cap = cv2.VideoCapture(1)
    start = time.time()
    while(cap.isOpened()):
        ## Capture frame-by-frame

        ret, cv2_im = cap.read()
        cv2.putText(cv2_im, piece_type_prediction, (int(cv2_im.shape[0]/2), int(cv2_im.shape[1]/3)), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 0 ,0), thickness = 4, lineType = 8)
        cv2.imshow('frame', cv2_im)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if (time.time() - start) > 1:
            cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)

            '''In order to correctly identify the image and maintain the aspect ratio,
            We crop the center 200x200 area and send it into predict function.'''

            pil_im = pim_im.crop((220, 140, 420, 340))
            try:
                piece_type_prediction = predict(model, pil_im, target_size)
            except:
                pass
            start = time.time()

    cap.release()
    cv2.destroyAllWindows()
