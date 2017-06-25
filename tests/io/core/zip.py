import unittest


class ZipTest(unittest.TestCase):
    def test_zip_minimal(self):
        """Minimal test with python lists as iterators."""
        from inferno.io.core import Zip
        from torch.utils.data.dataset import Dataset

        with self.assertRaises(AssertionError):
            zipped = Zip([1, 2, 3], [4, 5, 6, 7])

        # This is required because Zip checks if its inputs are actually torch datasets
        class ListDataset(list, Dataset):
            pass

        dataset_1 = ListDataset([1, 2, 3, 4])
        dataset_2 = ListDataset([5, 6, 7, 8, 9])
        zipped = Zip(dataset_1, dataset_2)
        self.assertEqual(len(zipped), 4)

        fetched = zipped[1]
        self.assertEqual(fetched, [2, 6])

        with self.assertRaises(AssertionError):
            fetched = zipped[4]

    def test_zip_sync(self):
        """Test synchronization mechanics."""
        # TODO


if __name__ == '__main__':
    unittest.main()
