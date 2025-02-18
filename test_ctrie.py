import unittest
import ctrie

class TestCtrie(unittest.TestCase):
    def setUp(self):
        self.index = ctrie.DictIndex(end_of_triple=0)
        self.tree1 = [1, [1,2,3,4]]
        self.tree2 = [2, [1,2,3,{4:[1,[1,2]],5:[1,[2,3]]}]]
        self.tree3 = [18, [150000, 366, 36, 3168, 301, 22703, 29, 366, {
                            8846: [1, [4096, 29, 366, 96924, 389, 279, 56690, 409, 21725, 304, 12366, 11, 9822, 29, 662]],
                            4789: [2, [29, 366, {
                                78678: [1, [7559, 389, 279, 56690, 409, 21725, 304, 12366, 11, 9822, 29, 662]],
                                34717: [1, [287, 555, 8563, 7462, 64, 359, 352, 304, 279, 80735, 939, 51984, 16730, 323, 83797, 554, 19558, 29, 662]]}]], # deforme!
                            34713: [1, [29, 366, 13020, 58618, 29, 662]],
                            4956: [1, [315, 29, 366, 34717, 287, 320, 30318, 29409, 11, 7479, 10255, 2740, 9960, 449, 6308, 16401, 662]],
                            33498: [1, [29, 366, 35632, 7462, 64, 359, 352, 29, 662]],
                            258: [1, [1010, 29, 366, 5926, 21, 12, 1721, 12, 1721, 51, 410, 25, 410, 25, 410, 57, 29, 662]],
                            28010: [2, [505, 3769, 29, 366, {
                                20538: [1, [320, 34717, 287, 7479, 1903, 315, 9193, 8987, 55618, 14733, 2695, 16651, 13354, 16401, 662]],
                                78009: [1, [6308, 29, 662]]}]],
                            13727: [1, [29, 366, 39, 404, 939, 51984, 16730, 323, 83797, 554, 19558, 29, 662]],
                            15237: [1, [31095, 29, 366, 36, 3168, 301, 22703, 29, 662]],
                            3902: [1, [3917, 29, 366, 36, 3168, 301, 22703, 29, 662]],
                            2627: [1, [29, 86803, 6330, 13, 21, 29, 662]],
                            3175: [1, [29, 86803, 4364, 29, 662]],
                            2588: [1, [29, 366, 39, 404, 939, 51984, 16730, 323, 83797, 554, 19558, 29, 662]],
                            15859: [1, [2704, 29, 366, 12965, 8106, 29, 662]],
                            11389: [1, [29, 366, 23175, 4273, 29, 662]],
                            4581: [1, [315, 279, 4101, 29, 366, 36, 3168, 301, 22703, 320, 35, 8458, 359, 352, 4101, 16401, 662]]}]]

    def test_count_leaves(self):
        self.assertEqual(self.index.count_leaves(self.tree1), 1)
        self.assertEqual(self.index.count_leaves(self.tree2), 2)
        self.assertEqual(self.index.count_leaves(self.tree3), 18)

    def test_merge(self):
        numleaves1 = self.index.merge(self.tree1)
        other1 = ctrie.DictIndex(end_of_triple=0, tree=self.tree1)
        self.assertEqual(self.index, other1)
        self.assertEqual(numleaves1, 1)

        numleaves2 = self.index.merge(self.tree2)
        other2 = ctrie.DictIndex(end_of_triple=0, tree=self.tree2)
        self.assertEqual(self.index, other2)
        self.assertEqual(numleaves2, 2)

        numleaves3 = self.index.merge(self.tree3)
        self.assertEqual(numleaves3, 20)
        self.assertEqual(numleaves3, self.index.count_leaves())

        # new leaf
        numleaves4 = self.index.merge([1, [315, 279, 4101, 29, 366, 36, 3168, 301, 22703, 320, 35, 8458, 359, 352, 4101, 16401, 662]])
        self.assertEqual(numleaves4, 21)
        self.assertEqual(numleaves4, self.index.count_leaves())

        # existing leaf
        numleaves5 = self.index.merge([1, [150000, 366, 36, 3168, 301, 22703, 29, 366, 8846, 4096, 29, 366, 96924, 389, 279, 56690, 409, 21725, 304, 12366, 11, 9822, 29, 662]])
        self.assertEqual(numleaves5, 21)
        self.assertEqual(numleaves5, self.index.count_leaves())

if __name__ == "__main__":
    unittest.main()
