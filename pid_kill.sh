#특정프로세스 종료
kill -9 `ps -ef | grep python | grep KoSimCSE |awk '{print $2}'`