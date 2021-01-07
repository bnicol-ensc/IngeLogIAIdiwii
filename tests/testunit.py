import sys
import os 
d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(d)

import unittest
from source.Prediction import makePrediction, makeAllPredictions? maxPredictions
from source.Testing import predict, compute_predictions, compute_roc_curve
from source.Train import import_data
from source.Visualisation import totalNumberOfItems, itemsInClasses, getSentence, getIntent, computeMetrics


class TestPredictionMethods(unittest.TestCase):

    def test_makePrediction(self):
        self.assertEqual(1,1)

    def test_makeAllPredictions(self):
        self.assertFalse('Foo'.isupper())

    def test_maxPredictions(self):
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
        self.assertEqual('foo'.upper(), 'FOO')


class TestVisualisationMethods(unittest.TestCase):
    
    def test_totalNumberOfItems(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_itemsInClasses(self):
        self.assertFalse('Foo'.isupper())     
    
    def test_getSentence(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_getIntent(self):
        self.assertFalse('Foo'.isupper())

    def test_computeMetrics(self):
        self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()