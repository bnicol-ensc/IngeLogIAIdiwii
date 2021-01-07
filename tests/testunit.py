import sys
import os 
d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(d)

import unittest
from source.Prediction import *
from source.Testing import predict, compute_predictions, compute_roc_curve
from source.Train import *
from source.Visualisation import *


class TestPredictionMethods(unittest.TestCase):

    def test_function(self):
        self.assertEqual(1,1)


class TestTestingMethods(unittest.TestCase):
    
    def test_predict(self):
        self.assertTrue(len(predict())==2)

    def test_compute_predictions(self):
        self.assertEqual(len(compute_predictions()),3)
    
    def test_compute_roc_curve(self):
        self.assertEqual(len(compute_roc_curve()),3)


class TestTrainMethods(unittest.TestCase):
    
    def test_import_data(self):
        import_data
        self.assertEqual('foo'.upper(), 'FOO')
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())


class TestVisualisationMethods(unittest.TestCase):
    
    def test_function(self):
        self.assertEqual('foo'.upper(), 'FOO')
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())


if __name__ == '__main__':
    unittest.main()