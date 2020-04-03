import unittest
import inferno.utils.model_utils as mu
from inferno.utils.partial_cls import register_partial_cls
import torch
import torch.nn as nn


class TestCls(object):
    def __init__(self, a, b, c=1, d=2):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

class PartialClsTester(unittest.TestCase):

    def test_partial_cls(self):
        register_partial_cls(TestCls, 'TestA', 
            fix=dict(a='a'),
            default=dict(b='b'),
            module=__name__
        )
        assert 'TestA' in globals()

        inst = TestA()
        assert inst.a == 'a'
        assert inst.b == 'b'
        assert inst.c == 1
        assert inst.d == 2

        inst = TestA('fu','bar','fubar')
        assert inst.a == 'a'
        assert inst.b == 'fu'
        assert inst.c == 'bar'
        assert inst.d == 'fubar'

        with self.assertRaises(TypeError):
            inst = TestA(a=2)

    def test_update_existing_default_cls(self):
        register_partial_cls(TestCls, 'TestA', 
            fix=dict(a='a'),
            default=dict(d=3),
            module=__name__
        )
        assert 'TestA' in globals()

        inst = TestA(42)
        assert inst.a == 'a'
        assert inst.b == 42
        assert inst.c == 1
        assert inst.d == 3

        with self.assertRaises(TypeError):
            inst = TestA()

    def test_fix_nothing(self):
        register_partial_cls(TestCls, 'TestA',
            module=__name__
        )
        assert 'TestA' in globals()

        inst = TestA(1,2,3,4)
        assert inst.a == 1
        assert inst.b == 2
        assert inst.c == 3
        assert inst.d == 4

        with self.assertRaises(TypeError):
            inst = TestA()

    def test_fix_all(self):
        register_partial_cls(TestCls, 'TestA',
            module=__name__,
            fix=dict(a=4, b=3, c=2, d=1)
        )
        assert 'TestA' in globals()

        inst = TestA()
        assert inst.a == 4
        assert inst.b == 3
        assert inst.c == 2
        assert inst.d == 1

        with self.assertRaises(TypeError):
            inst = TestA('a')

        with self.assertRaises(TypeError):
            inst = TestA(a=1)
        with self.assertRaises(TypeError):
            inst = TestA(b=1)
        with self.assertRaises(TypeError):
            inst = TestA(c=1)
        with self.assertRaises(TypeError):
            inst = TestA(d=1)


    def test_default_all(self):
        register_partial_cls(TestCls, 'TestA',
            module=__name__,
            default=dict(a=4, b=3, c=2, d=1)
        )
        assert 'TestA' in globals()

        inst = TestA()
        assert inst.a == 4
        assert inst.b == 3
        assert inst.c == 2
        assert inst.d == 1


        inst = TestA(2)
        assert inst.a == 2
        assert inst.b == 3
        assert inst.c == 2
        assert inst.d == 1

        inst = TestA(2,3,4,5)
        assert inst.a == 2
        assert inst.b == 3
        assert inst.c == 4
        assert inst.d == 5

        with self.assertRaises(TypeError):
            inst = TestA(3,4,5,a=2)
            
        inst = TestA(3,4,5,d=2)
        assert inst.a == 3
        assert inst.b == 4
        assert inst.c == 5
        assert inst.d == 2


      

if __name__ == '__main__':
    unittest.main()
