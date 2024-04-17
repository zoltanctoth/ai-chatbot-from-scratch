FROM ubuntu
RUN apt-get update
RUN apt-get install -y python3-pip build-essential

WORKDIR /

COPY requirements.txt .

RUN pip install  -r requirements.txt
RUN pip3 uninstall -y ctransformers
RUN pip3 install --force ctransformers --no-binary ctransformers

COPY llama2.py .
COPY chainlit-llama2.md chainlit.md

# Download the model
RUN python3 llama2.py 

# Test run
RUN python3 llama2.py test

EXPOSE 80
CMD ["chainlit", "run", "llama2.py", "--host", "0.0.0.0", "--port", "80", "--headless"]
