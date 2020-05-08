from flow.data_prepare import get_pkl, exist_pkl, print_
from custom import multi_prepare_record, multi_generate_record
from keras.models import load_model
import cv2


def infer(args):
    if args.image is None and args.images is None:
        assert exist_pkl(args.base)
        print_("use the existing pkl file")
        train_collection, test_collection = get_pkl(args.base)
        model = load_model('')
        for e in test_collection:
            e = multi_prepare_record(e)
            x, y = multi_generate_record(e, False)
            predicted = model.predcit()
            print(y, predicted)
            img = x.squeeze()
            cv2.imshow('img', img)
            cv2.waitKey()
