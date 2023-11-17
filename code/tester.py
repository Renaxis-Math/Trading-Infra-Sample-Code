# This file contains all test suites
# for all functions that return thing in the project

# May require module installation
# Can run file directly or use pytest tester.py under the correct directory

# import coverage   // if needed
import pandas as pd
import numpy as np
import importlib
import helper, consts

importlib.reload(helper)
importlib.reload(consts)

class TestSuite():
    def run(self):
        self.test_build_feature_map()
        self.test_get_df_with_interaction_terms()
    
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
        
    def test_get_df_with_interaction_terms(self):
        df = pd.DataFrame({'A':[1,2,3], 'B':[4,5,6], 'C':[7,8,9], 
                           'D': [10, 11, 12], 'E':[13, 14, 15]})
        
        col_pairs = [
            ['A', 'D'],
            ['C', 'B', 'E']
        ]
        new_df = helper.get_df_with_interaction_terms(df, col_pairs)
        #assert all(new_df[('A', 'D')] == df['A'] * df['D']), print(new_df)
        #assert all(new_df[('C', 'B', 'E')] == df['C'] * df['B'] * df['E']), print(new_df)
    
    def test_get_train_from_testday(self):
        testday1 = '20140501'
        testday2 = '20180101'
        train_1_start, train_1_end = helper.get_train_from_testday(testday1)
        train_2_start, train_2_end = helper.get_train_from_testday(testday2)

        assert((train_1_start == "20130301"))
        assert((train_1_end == "20130331"))
        assert((train_2_start == "20161101"))
        assert((train_2_end == "20131231"))


test_suite = TestSuite()
test_suite.run()