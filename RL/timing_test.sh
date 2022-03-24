##################
#  TIC-TOC       #
##################
#  DESCRIPTION
#  Times the execution of a script in a log file. You need to enclose the commands between tic and toc.

# USAGE:
# tic.sh  <LOG_FILE> 
# scripts_to_be_executed.sh
# toc.sh <LOG_FILE>


# DETAILED DESCRIPTION:
# Chains the following steps (please see pre_process.py for reference)

# date +%s > $1

# a=$(date +%s)
# echo $a

# echo sum
# echo $(($a + $a))

# echo "scale=2 ; $a / 60" | bc

# echo "scale=2 ; 10 / 60" | bc

# total_min=$(echo "scale=2 ; 10 / 60" | bc)

tic.sh /workspace/garbage/logtic.log
cat -dsfsdf 4
toc.sh /workspace/garbage/logtic.log

