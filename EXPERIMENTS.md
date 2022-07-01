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
    python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/seq_replay_baseline/parameters/seq_replay_baseline_n_col=4_n_batches=100_T=1 -c=plot_figs_1c_6b
    ```

  - #### Figure 1D
    ```commandline
    python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/seq_replay_baseline/parameters/seq_replay_baseline_n_col=4_n_batches=100_T=1 -c=plot_fig1_d
    ```

- ### Learning accuracy
  Running the simulations:
    ```commandline
    python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/parameters/seq_replay_4x700.py -c=seq_proc
    ```

  Data will be stored under `cone_shouval_2021/data/seq_replay_baseline_4x700/`. After running the plotting scripts below,
  the figures will be stored in the `/figures` subdirectory.


  - #### Figure 2A
      ```commandline
      python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/seq_replay_baseline_4x700/parameters/seq_replay_baseline_4x700_n_col=4_n_batches=100_T=1 -c=plot_fig2_a
      ```

  - #### Figure 2B
      ```commandline
      python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/seq_replay_baseline_4x700/parameters/seq_replay_baseline_4x700_n_col=4_n_batches=100_T=1 -c=plot_fig2_b
      ```

  - #### Figure 2C
      ```commandline
      python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/seq_replay_baseline_4x700/parameters/seq_replay_baseline_4x700_n_col=4_n_batches=100_T=1 -c=plot_fig2_c
      ```


- ### Robustness tests

  #### 1. Variations of the intra-columnar weights
  Running the simulations:
    ```commandline
    python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/parameters/seq_robustness_fixed.py -c=seq_proc
    ```

  Data will be stored under `cone_shouval_2021/data/robustness_fixed_4x700/`. After running the plotting scripts below,
  the figures will be stored in the `/figures` subdirectory.


  - #### Figure 3A and Figure 3B (left panels)
      ```commandline
      python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/robustness_fixed_4x700/parameters/robustness_fixed_4x700_n_col=4_n_batches=100_T=1 -c=plot_fig3_a
      ```

  #### 2. Variations of the feed-forward inhibitory weights

  Running the simulations:
  ```commandline
  python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/parameters/seq_robustness_fixed_ff.py -c=seq_proc
  ```

  Data will be stored under `cone_shouval_2021/data/robustness_fixed_ff_4x700/`. After running the plotting scripts below,
  the figures will be stored in the `/figures` subdirectory.


  - #### Figure 3B (right panels)
      ```commandline
      python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/robustness_fixed_ff_4x700/parameters/robustness_fixed_ff_4x700_n_col=4_n_batches=100_T=1 -c=plot_fig3_b
      ```

  #### 3. Variations in the learning parameters

  Running the simulations (separately for randomization per instance and per trial):
  ```commandline
  python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/parameters/seq_robustness_randomized_instance.py -c=seq_proc
  python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/parameters/seq_robustness_randomized_trial.py -c=seq_proc
  ```

  Data will be stored under `cone_shouval_2021/data/robustness_learn_rand_4x700_inst/` and  `cone_shouval_2021/data/robustness_learn_rand_4x700_trial/`. After running the plotting scripts below,
  the figures will be stored in the `/figures` subdirectory.
  Note that we assume that the simulations of the baseline network have also been completed (but this requirement can be relaxed by editing the plotting code).

 
  - #### Figure 3C 
      ```commandline     
      python execute.py --project=cone_shouval_2021 -p=cone_shouval_2021/data/robustness_learn_rand_4x700_trial/parameters/robustness_learn_rand_4x700_trial_n_col=4_distrange=20.0_T=1 -c=plot_fig3_c \ 
          --extra rand1_param_file=cone_shouval_2021/data/robustness_learn_rand_4x700_inst/parameters/robustness_learn_rand_4x700_inst_n_col=4_T=1 \ 
                  baseline_param_file=cone_shouval_2021/data/seq_replay_baseline/parameters/seq_replay_baseline_n_col=4_n_batches=100_T=1
      ```

  #### 4. Variations in the T -> I_T (L5) synaptic strengths 

  Running the simulations:
  ```commandline
  python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/parameters/seq_robustness_inh_l5.py -c=seq_proc
  ```

  Data will be stored under `cone_shouval_2021/data/robustness_IL5_3x700/`. After running the plotting scripts below,
  the figures will be stored in the `/figures` subdirectory.


  - #### Figure 4A
      ```commandline
      python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/robustness_IL5_3x700/ParameterSpace.py -c=plot_fig4_a
      ```


- ### Model scaling
  #### 1. Standard scaling
  Running the simulations:
    ```commandline
    python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/parameters/scale_standard.py -c=seq_proc
    ```

  Data will be stored under `cone_shouval_2021/data/scale_standard/`. After running the plotting scripts below,
  the figures will be stored in the `/figures` subdirectory.


  - #### Figure 5A
      ```commandline
      python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/scale_standard/parameters/scale_standard_T=1 -c=plot_figs_4bcd_5ab_6de --extra plot_epoch=0 plot_batch=40 train_set=True
      python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/scale_standard/parameters/scale_standard_T=1 -c=plot_figs_4bcd_5ab_6de --extra plot_epoch=test_epoch plot_batch=test_batch_1 train_set=False   
      ```


  #### 2. Manual scaling
  Running the simulations:
    ```commandline
    python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/parameters/scale_manual.py -c=seq_proc
    ```

  Data will be stored under `cone_shouval_2021/data/scale_manual/`. After running the plotting scripts below,
  the figures will be stored in the `/figures` subdirectory.


  - #### Figure 5B
      ```commandline
      python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/scale_manual/parameters/scale_manual_p1=0.2_p2=0.2_T=1 -c=plot_figs_4bcd_5ab_6de --extra plot_epoch=0 plot_batch=90 train_set=True
      python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/scale_manual/parameters/scale_manual_p1=0.2_p2=0.2_T=1 -c=plot_figs_4bcd_5ab_6de --extra plot_epoch=test_epoch plot_batch=test_batch_1 train_set=False   
      ```

  - ### Projections between all columns

    #### 1. No parameter changes

    Running the simulations:
      ```commandline
      python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/parameters/seq_replay_ff_all.py -c=seq_proc
      ```

    Data will be stored under `cone_shouval_2021/data/seq_replay_ff_all_4x700/`. After running the plotting scripts below,
    the figures will be stored in the `/figures` subdirectory.

    - #### Figure 6B
        ```commandline
        python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/seq_replay_ff_all_4x700/parameters/seq_replay_ff_all_4x700_n_col=4_n_batches=100_T=1 -c=plot_fig6_b
        ```

  - #### Figure 6C
      ```commandline
      python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/seq_replay_ff_all_4x700/parameters/seq_replay_ff_all_4x700_n_col=4_n_batches=100_T=1 -c=plot_fig6_c
      ```

    #### 2. Low background noise

    Running the simulations:
      ```commandline
      python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/parameters/seq_replay_ff_all_low_noise.py -c=seq_proc
      ```

    Data will be stored under `cone_shouval_2021/data/seq_replay_ff_all_4x700_lowBgNoise_w/`. After running the plotting scripts below,
    the figures will be stored in the `/figures` subdirectory.

  - #### Figure 6D
    ```commandline
    python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/seq_replay_ff_all_4x700_lowBgNoise_w/parameters/seq_replay_ff_all_4x700_lowBgNoise_w_n_col=4_wIEL5rec=0.21_T=1 -c=plot_figs_4bcd_5ab_6de --extra plot_epoch=test_epoch plot_batch=test_batch_1 train_set=False
    ```
    

  #### 3. High Hebbian threshold for the FF connections

  Running the simulations:
    ```commandline
    python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/parameters/seq_replay_ff_all_high_th.py -c=seq_proc
    ```

  Data will be stored under `cone_shouval_2021/data/seq_replay_ff_all_high_th/`. After running the plotting scripts below,
  the figures will be stored in the `/figures` subdirectory.

  - #### Figure 6E
    ```commandline
    python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/seq_replay_ff_all_high_th/parameters/seq_replay_ff_all_high_th_n_col=4_th=0.03_T=1 -c=plot_figs_4bcd_5ab_6de --extra plot_epoch=test_epoch plot_batch=test_batch_1 train_set=False
    ```

  - ### Alternative wiring with local inhibition 
  
    Running the simulations:
      ```commandline
      python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/parameters/seq_replay_local_inh.py -c=seq_proc
      ```

    Data will be stored under `cone_shouval_2021/data/alt_model_local_inh/`. After running the plotting scripts below,
    the figures will be stored in the `/figures` subdirectory.

    - #### Figure 7B
        ```commandline
        python execute.py --project=cone_shouval_2021 -p=./cone_shouval_2021/data/alt_model_local_inh/parameters/alt_model_local_inh_w_IE_L23_L5=0.2_T=1 -c=plot_fig7_B
        ```

