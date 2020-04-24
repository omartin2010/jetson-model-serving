
## Model Serving
Serving models from Jetson TX2
Flask app to score models for a jetson tx2. Models built using TF object detection API.

Details on this are on another project related project.
```
docker run --restart unless-stopped --privileged -d \
    --name modelserving \
    --memory 2g \
    --memory-swap 3g \
    --network host \
    modelserving:lowmem
```