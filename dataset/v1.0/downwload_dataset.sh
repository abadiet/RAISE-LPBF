#!bin/bash

nohup curl -L "https://uftp.fz-juelich.de:9112/UFTP_Auth/rest/access/JUDAC/0a4013b9-d138-4085-a03b-2b91a1e96cb9/RAISE_LPBF_train.hdf5" -o "RAISE_LPBF_train.hdf5" &> nohup_train.out &
nohup curl -L "https://uftp.fz-juelich.de:9112/UFTP_Auth/rest/access/JUDAC/0a4013b9-d138-4085-a03b-2b91a1e96cb9/RAISE_LPBF_test.hdf5" -o "RAISE_LPBF_test.hdf5" &> nohup_test.out &
