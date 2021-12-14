# StyleGAN3 VJ


```bash
docker build . -t `whoami`_vj:1.0 -f Dockerfile

docker run --runtime=nvidia -it \
 --rm -v /home/ryo.okada/stylegan3-vj:/workspace \
 --name `whoami`_svj `whoami`_vj:1.0 /bin/bash
```