from keras.models import model_from_yaml
from support_functions import data_load, predict_sequences_multiple, plot_results_multiple

# load the model data from training
data = 'data/daily_sp500/daily/table_nvda.csv'
input_arr = [50, 0.33]
# load the data
X_test, X_train, y_test, y_train, max = data_load(data, input_arr)

# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights("model_weights.h5")
print("Loaded model from disk")

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
score = model.evaluate(X_train, y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# plot the predictions
predictions = predict_sequences_multiple(model, X_test, 50, 50)
plot_results_multiple(predictions, y_test, 50)
