import unittest
import ctrie

class TestCtrie(unittest.TestCase):
    def setUp(self):
        self.index = ctrie.DictIndex(end_of_triple=0)
        self.tree1 = [1, [1,2,3,4]]
        self.tree2 = [2, [1,2,3,{4:[1,[1,2]],5:[1,[2,3]]}]]

    def test_count_leaves(self):
        self.assertEqual(self.index.count_leaves(self.tree1), 1)
        self.assertEqual(self.index.count_leaves(self.tree2), 2)

    def test_merge(self):
        numleaves1 = self.index.merge(self.tree1)
        other1 = ctrie.DictIndex(end_of_triple=0, tree=self.tree1)
        self.assertEqual(self.index, other1)
        self.assertEqual(numleaves1, 1)

        numleaves2 = self.index.merge(self.tree2)
        other2 = ctrie.DictIndex(end_of_triple=0, tree=self.tree2)
        self.assertEqual(self.index, other2)
        self.assertEqual(numleaves2, 2)

if __name__ == "__main__":
    unittest.main()
