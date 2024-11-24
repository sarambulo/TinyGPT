import unittest
from model import Tiny
import torch

class TestModelInitialization(unittest.TestCase):

    def test_call(self):
        model = Tiny(vocab_size=60)
        batch = torch.zeros((1,8), dtype=int)
        X_train = batch[:, :-1]
        y_train_true = batch[:, 1:]
        y_hat, loss = model(X_train, y_train_true)
        # TODO: Check the batch dim (it should be 1, 7, 60),
        # it is being folded into the first dim right now
        self.assertEqual((7, 60), y_hat.shape)
        self.assertTrue(torch.is_tensor(loss))

if __name__ == '__main__':
    unittest.main()