from probe_subproblem_transfer import indexed_root_shard


def test_indexed_root_shards_are_disjoint_and_complete() -> None:
    roots = [f"root-{index}" for index in range(7)]
    shard_zero = indexed_root_shard(roots, num_shards=2, shard_index=0)
    shard_one = indexed_root_shard(roots, num_shards=2, shard_index=1)

    assert shard_zero == [(0, "root-0"), (2, "root-2"), (4, "root-4"), (6, "root-6")]
    assert shard_one == [(1, "root-1"), (3, "root-3"), (5, "root-5")]
    assert sorted(shard_zero + shard_one) == list(enumerate(roots))
