# get the required modules
import cnn

# create the CNN model
model = cnn.CNN()

# train the model
model.train()

# save the model
model.save_model()
