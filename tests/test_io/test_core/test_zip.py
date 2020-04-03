import unittest


class ZipTest(unittest.TestCase):
    def test_zip_minimal(self):
        """Minimal test with python lists as iterators."""
        from inferno.io.core import Zip
        from torch.utils.data.dataset import Dataset

        with self.assertRaises(TypeError):
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

        with self.assertRaises(IndexError):
            fetched = zipped[4]

    def test_zip_sync(self):
        """Test synchronization mechanics."""
        # TODO

    def test_zip_reject(self):
        from inferno.io.core import ZipReject
        from torch.utils.data.dataset import Dataset

        # This is required because Zip checks if its inputs are actually torch datasets
        class ListDataset(list, Dataset):
            pass

        def rejection_criterion(sample_1, sample_2):
            return sample_1 < sample_2

        dataset_1 = ListDataset([1, 2, 3, 4])
        dataset_2 = ListDataset([2, 1, 3, 4])
        dataset_3 = ListDataset([0, 1, 2, 3])

        zipped = ZipReject(dataset_1, dataset_2, dataset_3,
                           rejection_criterion=rejection_criterion,
                           random_jump_after_reject=False,
                           rejection_dataset_indices=[0, 1])
        fetched = zipped[0]
        self.assertSequenceEqual(fetched, [2, 1, 1])

        zipped = ZipReject(dataset_1, dataset_2, dataset_3,
                           rejection_criterion=rejection_criterion,
                           rejection_dataset_indices=[1, 0])
        fetched = zipped[0]
        self.assertSequenceEqual(fetched, [1, 2, 0])


if __name__ == '__main__':
    unittest.main()
