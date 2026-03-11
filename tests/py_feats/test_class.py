def test_class_default_variables():
    class Foo:
        foo: int = 123

        def __init__(self, val: int):
            self.foo = val

    f = Foo(234)
    assert f.foo == 234
    assert Foo.foo == 123
    f = Foo(345)
    assert f.foo == 345
    assert Foo.foo == 123
