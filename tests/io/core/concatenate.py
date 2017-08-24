import unittest


class ConcatenateTest(unittest.TestCase):
    def test_concatenate(self):
        from inferno.io.core import Concatenate
        from torch.utils.data.dataset import Dataset

        with self.assertRaises(AssertionError):
            cated = Concatenate([1, 2, 3], [4, 5, 6, 7])

        class ListDataset(list, Dataset):
            pass

        dataset_1 = ListDataset([1, 2, 3, 4])
        dataset_2 = ListDataset([5, 6, 7])
        dataset_3 = ListDataset([8, 9, 10, 11, 12])

        cated = Concatenate(dataset_1, dataset_2, dataset_3)
        self.assertEqual(len(cated), 12)

        # Try to fetch
        self.assertEqual(cated[2], 3)
        self.assertEqual(cated[4], 5)
        self.assertEqual(cated[6], 7)
        self.assertEqual(cated[10], 11)
        self.assertEqual(cated[11], 12)

        with self.assertRaises(AssertionError):
            _ = cated[12]

if __name__ == '__main__':
    unittest.main()
