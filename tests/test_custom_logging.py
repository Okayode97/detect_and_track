import os
import json
import pytest
from src.detector.custom_logging import log_results, incremental_average


class TestLogResults:
    def test_if_log_file_does_not_exist_file_is_created(self):
        data = {"elasped_time": 60.0, "fps": 0.017}
        assert not os.path.exists("test_log.json")
        log_results(data, "test", "test_model")
        assert os.path.exists("test_log.json")
        os.unlink("test_log.json")

    def test_new_logged_data_is_appended_to_created_file(self):
        data = {"elasped_time": 30.0, "fps": 0.03}
        log_results(data, "test", "test_model")
        
        with open("test_log.json", "r") as f:
            logged_data = json.load(f)
        
        expected_logged_data = {"test_model": {"elasped_time": {"avg": 30.0, "count": 1}, "fps": {"avg": 0.03, "count": 1}}}
        assert expected_logged_data == logged_data    
        os.unlink("test_log.json")

    def test_logged_metrics_are_aggregated_with_multiple_calls(self):
        data = {"elasped_time": 60.0, "fps": 0.017}
        log_results(data, "test", "test_model")

        data = {"elasped_time": 65.0, "fps": 0.015}
        log_results(data, "test", "test_model")

        data = {"elasped_time": 10.0, "fps": 0.1}
        log_results(data, "test", "test_model_2")
        
        with open("test_log.json", "r") as f:
            logged_data = json.load(f)
        
        expected_logged_data = {"test_model": {"elasped_time": {"avg": 62.5, "count": 2},
                                               "fps": {"avg": 0.016, "count": 2}},
                                "test_model_2": {"elasped_time": {"avg": 10.0, "count": 1},
                                                 "fps": {"avg": 0.1, "count": 1}}
                                }

        assert expected_logged_data == logged_data
        os.unlink("test_log.json")

    def test_new_metrics_can_be_added_after_several_calls(self):
        data = {"elasped_time": 60.0, "fps": 0.017}
        log_results(data, "test", "test_model")

        data = {"elasped_time": 65.0, "fps": 0.015, "cpu_useage": 2.0}
        log_results(data, "test", "test_model")

        with open("test_log.json", "r") as f:
            logged_data = json.load(f)
        
        expected_logged_data = {"test_model": {"elasped_time": {"avg": 62.5, "count": 2},
                                            "fps": {"avg": 0.016, "count": 2},
                                            "cpu_useage": {"avg": 2.0, "count": 1}}
                                }

        assert expected_logged_data == logged_data
        os.unlink("test_log.json")


class TestIncrementalAverage:
    @pytest.mark.parametrize("data, new_val", (([0, 1, 1, 2, 3, 5, 8], 13),
                                               ([1, 2, 3, 4], 5)))
    def test_average_correctly_recalculated(self, data, new_val):
        avg = sum(data)/len(data)

        # update list
        data.append(new_val)
        new_avg = sum(data)/len(data)

        assert new_avg == incremental_average(avg, new_val, len(data))