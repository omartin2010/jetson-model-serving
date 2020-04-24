
## Model Serving
Serving models from Jetson TX2
Flask app to score models for a jetson tx2. Models built using TF object detection API.

Details on this are on another project related project.
```
docker run --restart unless-stopped --privileged -d \
    --name modelserving \
    --memory 1.5g \
    --network host \
    modelserving:latest
```