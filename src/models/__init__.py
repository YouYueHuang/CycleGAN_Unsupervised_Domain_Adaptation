
import importlib
from models.base_model import BaseModel

def import_class_by_name(modelName, className, classType = BaseModel):
    # Given the model name and class name
    # the class in model which belongs to class type will be imported.
    modellib = importlib.import_module(modelName)

    className = className.replace('_', '')

    model = None
    for name, cls in vars(modellib).items():
        if name.lower() == className.lower() and issubclass(cls, classType):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of {} with class name that matches %s in lowercase." % (modelName, classType, className))
        exit(0)

    return model

def createModel(opt):
    modelName = "models.{}_model".format(opt.model)
    className = "{}model".format(opt.model)
    model = import_class_by_name( modelName, className)
    instance = model(opt)
    return instance
