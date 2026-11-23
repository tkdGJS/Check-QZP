#fio test nomjobs 8, 16, 32, 64, 128
#>>>fio_test_normal.sh 1 8 cervical randrw 
fio \
--name=$1_test \
--ioengine=libaio \
--directory=./fio_test/fio_dump \
--runtime=5000 \
--bs=4k \
--size=1G \
--direct=1 \
--time_based=1 \
--randrepeat=0 \
--norandommap=1 \
--thread \
--group_reporting \
--per_job_logs=0 \
--allow_mounted_write=1 \
--numjobs=$1 \
--rw=randrw 
echo "FIO TEST COMPLETE"

exit 0

#----------------------------------------------------------------

