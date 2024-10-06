#!/bin/bash

bash compile_pcie.sh 1
bash run_pcie.sh 1
bash compile_pcie.sh 0
bash run_pcie.sh 0
bash parse4.sh
