# d2o
## Easy steps for efficient implementations

- set up environment: 

```
pip install -r requirements.txt
```

```
conda create -n d2o python=3.10
conda activate d2o
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

```
cd lm-evaluation-harness
pip install -e .
cd ..
```

- For LM_eval_harness datasets (Generation):
```
CUDA_VISIBLE_DEVICES=1 python run_lm_eval_harness_generation.py --model_name_or_path meta-llama/Meta-Llama-3-8B \
            --tasks gsm8k \
            --cache_dir /fs/scratch/PAS2473/zhongwei_models \
            --use_real_merge True \
            --hh_ratio 0.05 \
            --recent_size 0.15 \
```

- For Long-bench :

```
####### inference #########
CUDA_VISIBLE_DEVICES=0 python run_pred_long_bench.py --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --cache_dir /fs/scratch/PAS2473/zhongwei_models \
    --use_real_merge True \
    --model_type llama3 \
    --hh_ratio 0.05 \
    --recent_ratio 0.15 \
    --action_name local_llama3_02 \
    --e True 
######### evluation ############

python eval_long_bench.py --model {MODEL} # MODEL is the dir name under pred/ Currently it support Llama family model and Mistral model.
```