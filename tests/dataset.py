import unittest
import random
from pymel.dataset import MamlMnist, MamlKMnist


class CVDSTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data = {
            "mnist" : {
                "#class" : 10,
                "imgs" : (28, 28),
                "ml" : MamlMnist(maml=True),
                "nml" : MamlMnist(maml=False)
            },
            "kmnist" : {
                "#class" : 10,
                "imgs" : (28, 28),
                "ml" : MamlKMnist(maml=True),
                "nml" : MamlKMnist(maml=False)
            }
        }
    
    def test_return(self):
        for dataset in self.data:
            ml_ds = self.data[dataset]['ml']
            nml_ds = self.data[dataset]['nml']
            
            ml_sample = ml_ds[random.randint(0, len(ml_ds))]
            
            self.assertIsInstance(ml_sample, dict)
            self.assertEqual(len(list(ml_sample.keys())), self.data[dataset]['#class'])

            sample_img = ml_sample[
                random.randint(0, self.data[dataset]['#class'])
            ]
            
            self.assertEqual(tuple(sample_img.shape), self.data[dataset]["imgs"])

if __name__ == '__main__':
    unittest.main()