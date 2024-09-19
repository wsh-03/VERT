#!/bin/bash

# benchmark_language='c'
# test_name='COUNT_POSSIBLE_PATHS_TOP_LEFT_BOTTOM_RIGHT_NXM_MATRIX_3'
# verification='bolero'
# python3 evaluation.py $benchmark_language $test_name $verification

python3 /home/wsh-v22/vert/torust.py c starcoder ~/test/c2rust_test/test_file 

python3 /home/wsh-v22/vert/evaluation.py c ADD_1_TO_A_GIVEN_NUMBER bolero

