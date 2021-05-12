from pydeeprecsys.rl.neural_networks.value_estimator import ValueEstimator
import numpy as np


def test_save_load(tmp_file_cleanup):
    # given a neural network
    network = ValueEstimator(4, [4], 1)
    # when we train it a little bit
    inputs = np.array([1, 2, 3, 4])
    output = 1
    for i in range(20):
        network.update(inputs, output)
    # then it can accurately make predictions
    predicted_value = network.predict(inputs).detach().cpu().numpy()[0]
    assert round(predicted_value) == output
    # and when we store params
    network.save(tmp_file_cleanup)
    # and recreate the network
    network = ValueEstimator(4, [4], 1)
    network.load(tmp_file_cleanup)
    # then the prediction is the same
    assert network.predict(inputs).detach().cpu().numpy()[0] == predicted_value
