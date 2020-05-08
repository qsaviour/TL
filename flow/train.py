from flow.data_prepare import get_train_test_collection
from custom import data_prepare, file_filter
from tool.others import name_decorator


@name_decorator
def train(args):
    """
    Just train.
    :param args: Just args.
    :return:
    """
    train_collection, test_collection = get_train_test_collection(args.base, data_prepare, file_filter,
                                                                  args.force)

    if args.parallel:
        from flow.data_generator import multiple_prepare, multiple_generator
        train_collection = multiple_prepare(train_collection, args.parallel)
        generator = multiple_generator(train_collection, args.batch, args.parallel, True)
        test_collection = multiple_prepare(test_collection, args.parallel)
        g_validate = multiple_generator(test_collection, args.batch, args.parallel, False, randomly=False)
    else:
        from flow.data_generator import single_generator
        generator = single_generator(train_collection, args.batch, True)
        g_validate = single_generator(test_collection, args.batch, False)

    from model import build_model
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    ckp = ModelCheckpoint('../stock/models/weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=5, min_lr=1e-6)
    model = build_model((128, 128, 3))
    print('fitting.....')

    model.fit_generator(generator, 300, 1000, callbacks=[ckp, reduce_lr], validation_data=g_validate,
                        validation_steps=1)


if __name__ == '__main__':
    train()
