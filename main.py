from data import get_dataset
from model import get_model, train_model, loadmodel, test_model

X_train, X_test, Y_train, Y_test = get_dataset('unsplash-images-collection/')
model, inception = get_model()
# model = train_model(model, inception, X_train)
model = loadmodel('final_model.h5')
test_model(model, inception, X_test)