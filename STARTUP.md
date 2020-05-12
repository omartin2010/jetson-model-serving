## Model Serving Startup
In order to start the daemon, you need to run this command; this will ensure that after reboots, it starts automatically.
```
docker run --restart unless-stopped --privileged -d \
    --name modelserving \
    --memory 1.5g \
    --network host \
    --log-driver json-file --log-opt max-size=500k \
    modelserving:lowmem
```