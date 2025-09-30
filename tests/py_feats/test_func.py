def test_func_with_args():
    def foo(*args) -> int:
        return len(list(args))

    assert foo() == 0
