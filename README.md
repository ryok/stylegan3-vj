# StyleGAN3 VJ


```bash
docker build . -t `whoami`_svj:2.0 -f Dockerfile

docker run --runtime=nvidia -it \
 --rm -v `pwd`:/workspace \
 --name `whoami`_svj `whoami`_svj:2.0 /bin/bash
```

```bash
python feature_extraction.py --bpm $bpm_of_music --path $path_to_mp3

wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

python gen_images.py --model_path $path_to_pretrained_model

python gen_images.py --model_path stylegan3-r-afhqv2-512x512.pkl

python create_video.py --music $path_to_mp3
```