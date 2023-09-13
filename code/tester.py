# This file contains all test suites
# for all functions that return thing in the project

# May require module installation
# Can run file directly or use pytest tester.py under the correct directory

# import coverage   // if needed
import importlib
import helper, consts

importlib.reload(helper)
importlib.reload(consts)

def test_build_feature_map():
    # Edge cases
    assert helper.build_feature_map("") == {}
    assert helper.build_feature_map("", ".txt") == {}
    assert helper.build_feature_map("test_input.") == {}
    assert helper.build_feature_map("test_input", "..txt") == {}
    assert helper.build_feature_map("test_input", "t.xt") == {}
    
    # Valid cases
    assert helper.build_feature_map("test_input", "txt") != {}
    assert helper.build_feature_map("test_input", ".txt") != {}
    assert helper.build_feature_map("test input", ".txt") != {}