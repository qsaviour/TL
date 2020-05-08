from flow.data_prepare import get_pkl, exist_pkl, print_
from custom import multi_prepare_record, multi_generate_record
from keras.models import load_model
import cv2
import platform
import numpy as np

annotation_g = ['tesla', 'others']


def infer(args):
    if args.image is None and args.images is None:
        assert exist_pkl(args.base)
        print_("use the existing pkl file")
        train_collection, test_collection = get_pkl(args.base)
        model = load_model('')
        for i, e in enumerate(test_collection):
            e = multi_prepare_record(e)
            x, y = multi_generate_record(e, False)
            predicted = model.predcit()
            print(y, predicted)
            img = x.squeeze()
            x, y, w, h = list(map(int, e['location']))
            if np.argmax(predicted[0]) != np.argmax(y[0]):
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.imwrite(img,
                        '../stock/view/{:05d}-{}-{}.png'.format(i, e['annotation'], annotation_g[np.argmax(predicted[0])]))
