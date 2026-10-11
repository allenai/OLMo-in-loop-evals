from olmo_eval.util import all_gather_object, get_world_size, is_distributed


def test_world_size_without_distributed_initialization():
    assert not is_distributed()
    assert get_world_size() == 1
    assert all_gather_object("local") == ["local"]
