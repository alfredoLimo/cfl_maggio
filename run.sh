#!/bin/bash

# This code is usually called from cross_validation.sh, and it starts the server and clients 
# for the federated learning process. The server is started first, and then the clients are started.

# import client_number from config.py
n_clients=$(python -c "from config import client_number; print(client_number)")

echo -e "\n\033[1;36mStarting server with model \033[0m\n"

python server.py &
sleep 15  # Sleep for 2s to give the server enough time to start

for i in $(seq 1 $n_clients); do
    echo "Starting client ID $i"
    python client.py --id "$i" &
done
 
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
# Clean up
echo "Shutting down - processes completed correctly"
trap - SIGTERM && kill -- -$$
