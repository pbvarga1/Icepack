#! /bin/csh -f

#echo ${0}
echo "running icepack.launch.csh"

source ./icepack.settings

set jobfile = $1

#==========================================

cat >> ${jobfile} << EOFR
./icepack "\${1}" >&! \$ICE_RUNLOG_FILE
EOFR

exit 0
