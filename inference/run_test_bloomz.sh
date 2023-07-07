echo ${PATH}
nvidia-smi

ROOTPATH=./TIM
InferCODEPATH=${ROOTPATH}/inference
OUTPATH=${InferCODEPATH}/wmt_bloomz7b1_lora_reward_wmt_hint_dict_nist_bz128_0.5
TestPATH=${ROOTPATH}/test_data/wmt22
rootmodel=${ROOTPATH}/bloomz-7b1-mt
modelpathroot=${ROOTPATH}/checkpoints/wmt_bloomz7b1_lora_reward_wmt_hint_dict_nist_bz128_0.5
vocabpath=${ROOTPATH}/vocab/en-cn.fl.txt

start_step=0
sep_step=500
end_step=0
SRC=zh
TGT=en

pyfile=infer_bloom.py

for step in `seq $start_step $sep_step $end_step`;
do
echo "step:" $step

modelpath=${modelpathroot} #/checkpoint-${step}
logfile=${OUTPATH}/log_step${step}

echo "modelpath" ${modelpath}

if [ ! -d ${OUTPATH} ] ; then
    mkdir -p ${OUTPATH}
    chmod 777 ${OUTPATH} -R
fi

tmp_dir=${OUTPATH}/tmp${SRC}2${TGT}

if [ ! -d $tmp_dir ] ; then
    mkdir -p $tmp_dir
    chmod 777 $tmp_dir -R
fi

GPU_NUM=6
END=`expr $GPU_NUM - 1`

test_file=newstest22.${SRC}-${TGT}.${SRC}
ref_file=newstest22.${SRC}-${TGT}.${TGT}

FILE_LINES=`wc -l ${TestPATH}/${test_file} | awk '{print $1}'`
EACH_LINE=`expr $FILE_LINES / $GPU_NUM + 1`
echo ${EACH_LINE}

split -l ${EACH_LINE} -d ${TestPATH}/${test_file} ${tmp_dir}/${test_file} &
wait

for j in $(seq 0 $END);
  do
    GPU_ID=`expr 0 + ${j}`
    echo ${GPU_ID}

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    python3 -u ${InferCODEPATH}/${pyfile} \
    -i ${tmp_dir}/${test_file}0${j} \
    -o ${tmp_dir}/${test_file}0${j}.out \
    --src=${SRC} --tgt=${TGT} \
    --rootmodel ${rootmodel} \
    --model_path $modelpath -l \
    --vocab ${vocabpath} \
    --reverse \
    &
#    --ifhint \
#    --ifsample \
#    --ifreranking \
#    --vocab ${vocabpath} \
#    --reverse \
#    &

done
wait

cat ${tmp_dir}/${test_file}*.out > ${OUTPATH}/${test_file}.trans

