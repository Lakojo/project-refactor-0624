FROM python:3.12-alpine

EXPOSE 5001/tcp
WORKDIR /app

RUN pip install pandas seaborn flask

COPY data.csv .
COPY app.py .

CMD [ "python", "./app.py" ]
