import unittest
from model import Tiny
from torch import tensor

class TestModelInitialization(unittest.TestCase):
    model = Tiny()

    def test_call(self):
        # self.assertEqual('foo'.upper(), 'FOO')
        batch = torch.tensor.zeros((1,8))
        X_train = batch[:, :-1]
        y_train_true = batch[:, 1:]
        model(X_train, y_train_true)

if __name__ == '__main__':
    unittest.main()