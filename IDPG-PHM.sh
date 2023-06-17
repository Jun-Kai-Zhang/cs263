ARCH=roberta_large

# for me, i set a same path
SAVE=/home/junkai/course/cs263/IDPG/checkpoints/
NEWSAVE=/home/junkai/course/cs263/IDPG/checkpoints/
ROBERTA_PATH=$SAVE'roberta_large_checkpoint.pt'
#ROBERTA_PATH=$SAVE'nli_large_checkpoint.pt'
#suffixlens="5"
#insertpositions=$1
#simply=$2
#LR="5e-4"

insertpositions="0"
suffixlen="5"
LRs="5e-4"
seeds="1"

pdim="16"
mode="1"
mkdir -p "main_results"
OUT_FILE='main_results/IDPG-PHM-p-layerb.txt'$pdim'-'$suffixlen

for LR in $LRs; do
    for insertposition in $insertpositions; do
        SUFFIX='-multi-phm-p-layerb-'$mode'-'$pdim'-'$suffixlen'-'$insertposition'-f'$LR'_5-'
        TASKs='bbq'
        for TASK in $TASKs; do
            for seed in $seeds; do
                node=0
                SAVE_FILE=$NEWSAVE$TASK$SUFFIX$seed
                bash 'scripts/multi-suffix-'$TASK'_finetune-phm-p-layerb.sh' $ROBERTA_PATH $SAVE_FILE $seed $pdim $node $suffixlen $insertposition $LR $mode
            done
            wait
            for seed in $seeds; do
                 SAVE_FILE=$NEWSAVE$TASK$SUFFIX$seed'/'
                 CUDA_VISIBLE_DEVICES=0 python 'scripts/'$TASK'_get_result.py' -i $SAVE_FILE -o $OUT_FILE -n $seed -t $insertposition -l $LR 
            done
            wait
           #SAVE_FILE=$NEWSAVE$TASK$SUFFIX
            #python 'scripts/bagging_'$TASK'.py' -i $SAVE_FILE -o $OUT_FILE
            #echo $TASK 'done'
        done
    done 
done
