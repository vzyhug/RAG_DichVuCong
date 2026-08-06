FROM ubuntu:latest
LABEL authors="duyhu"

ENTRYPOINT ["top", "-b"]