#!/bin/bash

for ((i=1; i<=5; i++))
do
    python run_analyse_es.py --dimension 10 --func_id $i --pop_size 10 --max_evals 1000 --threads 8 --max_iters 100
    python run_analyse_es.py --dimension 100 --func_id $i --pop_size 20 --max_evals 10000 --threads 8 --max_iters 500
    python run_analyse_es.py --dimension 1000 --func_id $i --pop_size 100 --max_evals 100000 --threads 8 --max_iters 1000
    python run_analyse_es.py --dimension 5000 --func_id $i --pop_size 100 --max_evals 1000000 --threads 8 --max_iters 10000
done
