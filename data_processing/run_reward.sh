#export LANG=C.UTF-8
#export LC_ALL=en_GB.UTF-8

datapath=${1:-/data/}
outdatapath=.
codepath=TIM/data_processing

THREAD_NUM=4

src="en"
tgt="zh"
srcfilename=wmt_${src}${tgt}.${src}
tgtfilename=wmt_${src}${tgt}.${tgt}
badfilename=wmt_${src}${tgt}.${src}.trans
outname=wmt_${src}${tgt}.trans.reward.json

if [[ "${src}" = "en" ]] && [[ "${tgt}" = "zh" ]] ;
then
  SRC="English"
  TGT="Chinese"
elif [[ "${src}" = "zh" ]] && [[ "${tgt}" = "en" ]] ;
then
  SRC="Chinese"
  TGT="English"
elif [[ "${src}" = "de" ]] && [[ "${tgt}" = "en" ]] ;
then
  SRC="German"
  TGT="English"
elif [[ "${src}" = "en" ]] && [[ "${tgt}" = "de" ]] ;
then
  SRC="English"
  TGT="German"
else
  exit
fi
tmp_dir=${datapath}/tmp

if [ ! -d $tmp_dir ]; then
    mkdir -p $tmp_dir
    chmod 777 $tmp_dir -R
fi

srcdata=${datapath}/${srcfilename}
tgtdata=${datapath}/${tgtfilename}
baddata=${datapath}/${badfilename}

FILE_LINES=`wc -l $srcdata | awk '{print $1}'`
EACH_LINE=`expr $FILE_LINES / $THREAD_NUM + 1`
echo ${EACH_LINE} #>> ${outdatapath}/log.${outname}

echo '#### Step 0 #### Spliting data ... ' >> ${outdatapath}/log.${outname}
split -l ${EACH_LINE} -d $srcdata ${tmp_dir}/$srcfilename &
split -l ${EACH_LINE} -d $tgtdata ${tmp_dir}/${tgtfilename}1 &
split -l ${EACH_LINE} -d $baddata ${tmp_dir}/${badfilename}2 &

wait
echo 'split done' #>> ${outdatapath}/log.${outname}

cd ${tmp_dir}
NUM=`ls -l |grep "^-"|wc -l`
echo ${NUM} >> ${outdatapath}/log.${outname}

python3=`which python3`

${python3} -u ${codepath}/generate_instruction_reward_data.py \
--input_data_path=${tmp_dir} \
--output_data_path=${outdatapath}/split_json \
-i ${srcfilename} ${tgtfilename}1 ${badfilename}2 \
-o=${outname} \
-p=${codepath}/prompt.json \
--src=${SRC} \
--tgt=${TGT} \
--thread_num=${NUM} \
-t="EN_NMT"

wait

echo "instruct data done" #>> ${outdatapath}/log.${outname}

rm -r ${tmp_dir}/*

#cd ${outdatapath}

#${python3} -u ${outdatapath}/cat_json_to_txt.py -i ${outdatapath}/split_json -o ${outdatapath} -f train.txt >> ${outdatapath}/log.cat_json
