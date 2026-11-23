echo 'test start'

for NUMJOBS in 64
do
	for filename in 8
	do


		rm -rf ./fio_test/fio_dump
		echo 'rm fio_dump'
		mkdir ./fio_test/fio_dump
		echo 'mk fio_dump' 
		rm -rf ./fio_test/cervical_checkpoint
		echo 'rm checkpoint'
		mkdir ./fio_test/cervical_checkpoint
		echo 'mk checkpoint'
		sleep 20


		#>>>fio_test_normal.sh 1 8 cervical randrw 
		fio \
		--name=$filename-$NUMJOBS-test \
		--ioengine=libaio \
		--directory=./fio_test/fio_dump \
		--runtime=5000 \
		--bs=4k \
		--size=1G \
		--direct=0 \
		--time_based=1 \
		--randrepeat=0 \
		--norandommap=1 \
		--thread \
		--group_reporting \
		--per_job_logs=0 \
		--allow_mounted_write=1 \
		--numjobs=$NUMJOBS \
		--rw=randrw & 
		PID=$!
		echo "FIO TEST COMPLETE"

		echo "$PID"
		sleep 1000
		echo "python start$filename-$NUMJOBS"
		python3 cervical_batch$filename.py 1> result_batch$filename-$NUMJOBS.txt
		echo "python end$filename-$NUMJOBS"
		echo "kill start $PID"
		kill -9 $PID
		echo "kill end$PID"
		sleep 120
	done	


done
