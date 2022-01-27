import os
import logging
import torch
import torch.nn as nn


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def load_model(ckpt_path, encoder=None, ode_func=None, classifier=None, device="cpu"):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")

    checkpt = torch.load(ckpt_path)
    if encoder is not None:
        encoder_state = checkpt["encoder"]
        encoder.load_state_dict(encoder_state)
        encoder.to(device)

    if ode_func is not None:
        ode_state = checkpt["ode"]
        ode_func.load_state_dict(ode_state)
        ode_func.to(device)

    if classifier is not None:
        classifier_state = checkpt["classifier"]
        classifier.load_state_dict(classifier_state)
        classifier.to(device)


def get_logger(logpath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()
