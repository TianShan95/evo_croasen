#!/bin/bash
python evaluator.py --subnet .tmp/iter_0/net_0_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_0_stats.txt --n_epochs 5 2>&1 | tee .tmp/iter_0/net_0_log.txt &
python evaluator.py --subnet .tmp/iter_0/net_1_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_1_stats.txt --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_2_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_2_stats.txt --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_3_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_3_stats.txt --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_4_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_4_stats.txt --n_epochs 5 2>&1 | tee .tmp/iter_0/net_4_log.txt &
python evaluator.py --subnet .tmp/iter_0/net_5_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_5_stats.txt --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_6_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_6_stats.txt --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_7_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_7_stats.txt --n_epochs 5 &
wait
python evaluator.py --subnet .tmp/iter_0/net_8_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_8_stats.txt --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_9_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_9_stats.txt --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_10_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_10_stats.txt --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_11_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_11_stats.txt --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_12_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_12_stats.txt --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_13_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_13_stats.txt --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_14_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_14_stats.txt --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_15_subnet.txt --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_15_stats.txt --n_epochs 5 &
wait
