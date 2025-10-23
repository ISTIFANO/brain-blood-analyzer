FROM python:3.10-slimx

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir streamlit ultralytics pillow opencv-python-headless

EXPOSE 8501

CMD ["streamlit", "run", "src/interfaces/CLI_Interface.py", "--server.port=8501", "--server.address=0.0.0.0"]
