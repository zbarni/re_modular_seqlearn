The variable `root_dir` will refer to the absolute path of the project root directory.

Switch to the source directory:

```commandline
cd src/
```

- ### Reproduction of original results

    Running the simulations:
    ```commandline
    python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/parameters/seq_replay_baseline.py -c=seq_proc
    ```
    
    Data will be stored under `cone_shouval_2021/data/seq_replay_baseline/`. After running the plotting scripts below,
    the figures will be stored in the `/figures` subdirectory.

  - #### Figure 1C 
    ```commandline
    python execute.py --project=rescience_cone_2021 -p=./rescience_cone_2021/data/seq_replay_baseline/parameters/seq_replay_baseline_n_col\=4_n_batches\=100_T\=1 -c=plot_figs_1c_6b
    ```

  - #### Figure 1D 
    ```commandline
    python execute.py --project=rescience_cone_2021 -p=./rescience_cone_2021/data/seq_replay_baseline/parameters/seq_replay_baseline_n_col\=4_n_batches\=100_T\=1 -c=TODO
    ```

Instructions for the other experiments will follow soon.