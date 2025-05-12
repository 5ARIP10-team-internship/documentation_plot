1. Add specific model in `load_model.py`. Like SAC, MBAC
2. Run `test_model`: Random reference in each episode with random seed.

   ``python test_model.py --model_type MAC --model_path models/MAC.pth --reward_function absolute --episodes 20 --ID 20eps``

3. Plot result: `plot_single.py` to plot test line plot into PDF and absolute error box figure. `plot_comparison.py`: Add data file and lable in the code and run it directly.
   
     ``python plot_single.py --csv test_results/MAC_1.csv --plot_type three_phase ``
