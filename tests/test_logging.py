import os
from src.detector.logging import log_results


def test_if_log_file_does_not_exist_file_is_created():
    data = {"metrics": {"elasped_time": 60.0, "fps": 20.0}}
    assert not os.path.exists("test_log.json")
    log_results(data, "test")
    assert os.path.exists("test_log.json")