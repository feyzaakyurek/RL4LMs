FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# install git
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y wget
RUN apt-get install unzip

# install java
RUN apt-get install -y openjdk-8-jdk
RUN apt-get install -y openjdk-8-jre
RUN update-alternatives --config java
RUN update-alternatives --config javac

WORKDIR /stage/
COPY rl4lms/envs/text_generation/caption_metrics/spice rl4lms/envs/text_generation/caption_metrics/spice

# other model downloads
WORKDIR /stage/rl4lms/envs/text_generation/caption_metrics/spice
RUN ./get_stanford_models.sh 
WORKDIR /stage/

# Copy the files to /stage
COPY setup.py ./
COPY requirements.txt ./

COPY rl4lms/ ./rl4lms
# finally install the package (with dependencies)
RUN pip install -e .

# download external models (since it requires dependencies)
RUN pip install markupsafe==2.0.1
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -m spacy download en_core_web_sm

COPY scripts/ ./scripts
COPY data ./data
COPY myscripts ./myscripts
COPY custom_reward.py ./
COPY myutil.py ./
COPY openai_key_me ./
COPY openai_key_der ./
COPY openai_key_ai2 ./
COPY openai_key_hub ./

# CMD ["sh", "./myscripts/run.sh"]
CMD ["sh", "./myscripts/run_scripting.sh"]
# CMD ["sh", "./myscripts/run_openai_summ.sh"]