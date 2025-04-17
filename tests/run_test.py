from test_io import TestNASStreamer
from test_processing import TestParallelProcessing


def run_tests():
    """Run all tests in the package."""
    test_classes = [TestNASStreamer, TestParallelProcessing]
    for test_class in test_classes:
        print(f"Running tests for {test_class.__name__}...")
        test_instance = test_class()
        test_instance.setup_method()
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                print(f"Running {method_name}...")
                getattr(test_instance, method_name)()
        test_instance.teardown_method()
    print("All tests completed.")


if __name__ == "__main__":
    run_tests()
