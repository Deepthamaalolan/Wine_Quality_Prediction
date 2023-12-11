FROM centos:7

RUN yum -y update && yum -y install python3 python3-pip java-1.8.0-openjdk wget unzip

RUN python3 -V

ENV PYSPARK_DRIVER_PYTHON python3
ENV PYSPARK_PYTHON python3

RUN pip3 install --upgrade pip
RUN pip3 install awscli
RUN pip3 install numpy pandas

WORKDIR /opt

# Download and extract Spark
RUN wget --no-verbose -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz" \
  && tar -xf apache-spark.tgz \
  && rm apache-spark.tgz \
  && ln -s spark-3.5.0-bin-hadoop3 spark

# Install necessary packages for mounting S3 (assuming your system is using yum)
RUN yum -y install fuse-overlayfs

ENV SPARK_HOME /opt/spark
ENV PATH $SPARK_HOME/bin:$PATH

RUN wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.8.0/aws-java-sdk-1.8.0.jar -P /spark/jars/
RUN wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.0.0/hadoop-aws-3.0.0.jar -P /spark/jars/

RUN mkdir /code

WORKDIR /code

# Copy source code
COPY /src/wine_quality.py /code/
COPY /src/ValidationDataset.csv /code/

ADD /src/bestmodal/ /code/

# Model will be fetched from S3 using IAM role associated with EMR instance
ENTRYPOINT ["/opt/spark/bin/spark-submit", "--packages", "org.apache.hadoop:hadoop-aws:2.7.3,com.amazonaws:aws-java-sdk:1.7.4", "wine_quality.py"]
