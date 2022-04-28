#!/bin/bash
python evaluator.py --subnet .tmp/iter_0/net_0.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_0.stats --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_1.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_1.stats --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_2.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_2.stats --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_3.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_3.stats --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_4.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_4.stats --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_5.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_5.stats --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_6.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_6.stats --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_7.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_7.stats --n_epochs 5 &
wait
python evaluator.py --subnet .tmp/iter_0/net_8.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_8.stats --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_9.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_9.stats --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_10.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_10.stats --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_11.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_11.stats --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_12.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_12.stats --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_13.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_13.stats --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_14.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_14.stats --n_epochs 5 &
python evaluator.py --subnet .tmp/iter_0/net_15.subnet --supernet_path ../experiment/evo_croasen/super_net/super_model_best.pth.tar --save .tmp/iter_0/net_15.stats --n_epochs 5 &
wait
