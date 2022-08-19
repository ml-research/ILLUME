FROM nvcr.io/nvidia/pytorch:21.09-py3
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "zsh"]
RUN ["apt-get", "install", "-y", "git"]
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install --upgrade jupyter
RUN pip install --upgrade ipywidgets
WORKDIR /workspace/repositories/ILLUME
ENV PYTHONPATH "${PYTHONPATH}:./"
RUN ["apt-get", "install", "-y", "parallel"]
RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "default-jre"]
