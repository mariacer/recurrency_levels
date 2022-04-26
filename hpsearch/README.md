# Hyperparameter searches

For running a hyperparameter search, use the [run_hpsearch](run_hpsearch.py) script, in combination with a config file. Use the [hpsearch.hpsearch_config](hpsearch/hpsearch_config.py) as template for the config file.
Run for example: 
```console
$ python3 run_hpsearch.py --out_dir=bio_rnn/out/hpsearch_audioset_rnn_microcircuit_anatomical --grid_module=bio_rnn.hpsearch_configs.audioset_rnn_microcircuit_anatomical --num_sample=200 --gpu_per_trial=0.5
```
You can run the `postprocess_hpsearch.py` script to make scatter plots of all hyperparameters over which you searched. 
Example command-line argument snippet: 
```console
$ python3 postprocess_hpsearch.py --out_dir=bio_rnn/out/hpsearch_audioset_rnn_microcircuit_random/2020-11-03_15-30-54 --performance_key=acc_test_last --grid_module=bio_rnn.hpsearch_configs.audioset_rnn_microcircuit_random --mode=max
```

The hyperparameter search script will automatically save a config file with the best hyperparameter configuration. You can run this config file directly via the `run_config.py` script. 
Example command-line argument snippet: 
```console
$ python3 run_config.py --config_module=bio_rnn.configs.audioset_rnn_microcircuit_random --dataset=audioset
```

If you want to look at preliminary results before the hyperparameter search is finished, you can run the `process_prelim_results.py` file:
```console
$ python3 process_prelim_results.py --result_dir=logs/hpsearch_directory
```
