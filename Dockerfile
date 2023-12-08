# FROM centos:7

# RUN yum -y update && yum -y install python3 python3-pip java-1.8.0-openjdk wget unzip

# RUN python3 -V

# ENV PYSPARK_DRIVER_PYTHON python3
# ENV PYSPARK_PYTHON python3

# RUN pip3 install --upgrade pip
# RUN pip3 install numpy pandas

# WORKDIR /opt

# # Download and extract Spark
# RUN wget --no-verbose -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz" \
#     && tar -xf apache-spark.tgz \
#     && rm apache-spark.tgz \
#     && ln -s spark-3.1.2-bin-hadoop3.2 spark

# # Install necessary packages for mounting S3 (assuming your system is using yum)


# ENV SPARK_HOME /opt/spark
# ENV PATH $SPARK_HOME/bin:$PATH


# # Copy source code
# COPY /src /code/src

# WORKDIR /code/

# # Model will be fetched from S3 using IAM role associated with EMR instance
# ENTRYPOINT ["/opt/spark/bin/spark-submit", "src/wine_quality.py"]

FROM centos:7

RUN yum -y update && yum -y install python3 python3-pip java-1.8.0-openjdk wget unzip

RUN python3 -V

ENV PYSPARK_DRIVER_PYTHON python3
ENV PYSPARK_PYTHON python3

RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas pyspark

WORKDIR /opt

# Download and extract Spark
RUN wget --no-verbose -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz" \
  && tar -xf apache-spark.tgz \
  && rm apache-spark.tgz \
  && ln -s spark-3.1.2-bin-hadoop3.2 spark

# Install necessary packages for mounting S3 (assuming your system is using yum)
RUN yum -y install fuse-overlayfs

# Configure S3FS mount
RUN echo "fs.s3fs.impl=org.apache.hadoop.fs.s3a.S3AFileSystem" >> /opt/spark/conf/core-site.xml

ENV SPARK_HOME /opt/spark
ENV PATH $SPARK_HOME/bin:$PATH

# Copy source code
COPY /src /code/src
COPY core-site.xml /opt/spark/conf/

WORKDIR /code/

# Model will be fetched from S3 using IAM role associated with EMR instance
ENTRYPOINT ["/opt/spark/bin/spark-submit", "src/wine_quality.py"]
