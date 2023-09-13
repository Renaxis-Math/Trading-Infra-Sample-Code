# This file contains all test suites
# for all functions that return thing in the project

# May require module installation
# Can run file directly or use pytest tester.py under the correct directory

# import coverage   // if needed
import importlib
import helper, consts

importlib.reload(helper)
importlib.reload(consts)

class TestSuite():
    def run(self):
        self.test_build_feature_map()
    
    def test_build_feature_map(self):
        # Edge cases
        assert helper.build_feature_map("") == {}
        assert helper.build_feature_map("", ".txt") == {}
        assert helper.build_feature_map("test_input") == {}
        assert helper.build_feature_map("test_input.") == {}
        assert helper.build_feature_map("test_input", "..txt") == {}
        assert helper.build_feature_map("test_input", "t.xt") == {}
        
        # File not found
        assert helper.build_feature_map("test_input", "txt") == {}
        assert helper.build_feature_map("test_input", ".txt") == {}
        assert helper.build_feature_map("test input.txt",) == {}
        assert helper.build_feature_map("test input", ".txt") == {}
        assert helper.build_feature_map("test_input.txt", "txt") == {}
        assert helper.build_feature_map("data_description.txt", "txt") == {}
        
test_suite = TestSuite()
test_suite.run()