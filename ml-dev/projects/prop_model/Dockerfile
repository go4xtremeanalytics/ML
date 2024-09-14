FROM python:3.9-slim

COPY . .
RUN pip install -r requirements.txt

# COPY app.py ./
# COPY dashboard.py ./

RUN python app.py

EXPOSE 9050
CMD ["python", "./dashboard.py"]