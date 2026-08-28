from monitor_gpu_idle_then_run import all_idle, parse_gpu_csv, selected_gpus


def test_parse_and_idle_thresholds() -> None:
    rows = parse_gpu_csv("0, 0, 19\n1, 5, 512\n")
    assert selected_gpus(rows, [0, 1]) == rows
    assert all_idle(rows, max_utilization=10, max_memory_mib=1024)
    rows[1]["utilization"] = 11
    assert not all_idle(rows, max_utilization=10, max_memory_mib=1024)


def test_any_busy_gpu_resets_cluster_idle() -> None:
    rows = parse_gpu_csv("0, 0, 19\n1, 0, 2048\n")
    assert not all_idle(rows, max_utilization=10, max_memory_mib=1024)
