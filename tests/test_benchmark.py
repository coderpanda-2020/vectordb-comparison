import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from benchmark import generate_data, calculate_recall, K

def test_generate_data():
    vectors = generate_data(100, 128)
    assert vectors.shape == (100, 128)
    assert vectors.dtype == 'float32'

def test_calculate_recall():
    ground_truth = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    predictions = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    recall = calculate_recall(ground_truth, predictions)
    assert recall == 1.0

    predictions_half = np.array([[1, 2, 3, 4, 5, 11, 12, 13, 14, 15]])
    recall_half = calculate_recall(ground_truth, predictions_half)
    assert recall_half == 0.5
