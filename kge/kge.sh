#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=kge
SAVE_PATH=models

#The first four parameters must be provided
MODE=$1
MODEL=$2
DATA_PATH=$3
GPU_DEVICE=$4
SAVE_ID=$5

SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"

#Only used in training
BATCH_SIZE=$6
NEGATIVE_SAMPLE_SIZE=$7
HIDDEN_DIM=$8
GAMMA=$9
ALPHA=${10}
LEARNING_RATE=${11}
MAX_STEPS=${12}
TEST_BATCH_SIZE=${13}

if [ $MODE == "train" ]
then

echo "Start Training......"

#CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_train \
#    --cuda \
#    --do_valid \
#    --do_test \
#    --data_path $DATA_PATH \
#    --model $MODEL \
#    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
#    -g $GAMMA -a $ALPHA -adv \
#    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
#    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
#    ${14} ${15} ${16} ${17} ${18} ${19} ${20}

python -u $CODE_PATH/run.py --do_train \
--do_test \
--data_path $DATA_PATH \
--model $MODEL \
-n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
-g $GAMMA -a $ALPHA -adv \
-lr $LEARNING_RATE --max_steps $MAX_STEPS \
-save $SAVE --test_batch_size $TEST_BATCH_SIZE \
${14} ${15} ${16} ${17} ${18} ${19} ${20}

elif [ $MODE == "valid" ]
then

echo "Start Evaluation on Valid Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_valid --cuda -init $SAVE

elif [ $MODE == "test" ]
then

echo "Start Evaluation on Test Data Set......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE

elif [ $MODE == "eval" ]
then

echo "Start Evaluation on Training Set and Rank the Scores......"

#CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_test --cuda -init $SAVE
python -u $CODE_PATH/run.py --do_eval -init $SAVE --train_path --noise_threshold  eliminate_noise_path

else
   echo "Unknown MODE" $MODE
fi