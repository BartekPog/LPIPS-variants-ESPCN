FROM python-cuda-12

RUN pip3 install virtualenv 
RUN virtualenv -p python3.12 /development/venv-lpips-variants
RUN /development/venv-lpips-variants/bin/pip install --upgrade pip
RUN /development/venv-lpips-variants/bin/pip install lpips==0.1.3

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN /development/venv-lpips-variants/bin/pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

COPY requirements.txt /development/requirements.txt
RUN /development/venv-lpips-variants/bin/pip install -r /development/requirements.txt