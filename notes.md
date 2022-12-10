# Getting Started

### Reference Documentation

For further reference, please consider the following sections:

* [Official Apache Maven documentation](https://maven.apache.org/guides/index.html)
* [Spring Boot Maven Plugin Reference Guide](https://docs.spring.io/spring-boot/docs/3.0.0/maven-plugin/reference/html/)
* [Create an OCI image](https://docs.spring.io/spring-boot/docs/3.0.0/maven-plugin/reference/html/#build-image)
* [Spring Boot DevTools](https://docs.spring.io/spring-boot/docs/3.0.0/reference/htmlsingle/#using.devtools)



### Projekt Doukmentation Forschritt

## Einrichtungsaufwand

minimal

# Umsetzung

Orientierung:

https://www.baeldung.com/java-cnn-deeplearning4j

---> Besseres Beispiel

https://github.com/deeplearning4j/deeplearning4j-examples/blob/165f406763330d5e7f8ce842e76d4376e24ff0d1/mvn-project-template/src/main/java/org/deeplearning4j/examples/sample/LeNetMNIST.java#L69

## Sonstige Anmekrungen: 

- keine Infos über Funktionen bei hovern über diese, bei djl viel besser 
- extrem langsames training meiner Beispiele im vergleich zu djl
- Schlechtes Memory Managment im Vergleich zu djl -> Bekomme Out of Memory Exception

// Prüfen ob an ubuntu liegt, das djl auf windows getestet

# Probleme udn Lösungen:

## Problem 1 mnist download:

2022-11-28T18:09:32.339+01:00  INFO 26196 --- [  restartedMain] c.e.d.Deeplearning4jCnnApplication       : Started Deeplearning4jCnnApplication in 1.006 seconds (process running for 2.042)
2022-11-28T18:09:32.339+01:00  INFO 26196 --- [  restartedMain] org.deeplearning4j.base.MnistFetcher     : Downloading MNIST...
Exception in thread "restartedMain" java.lang.reflect.InvocationTargetException
at java.base/jdk.internal.reflect.DirectMethodHandleAccessor.invoke(DirectMethodHandleAccessor.java:119)
at java.base/java.lang.reflect.Method.invoke(Method.java:577)
at org.springframework.boot.devtools.restart.RestartLauncher.run(RestartLauncher.java:49)
Caused by: java.net.UnknownHostException: benchmark.deeplearn.online
at java.base/sun.nio.ch.NioSocketImpl.connect(NioSocketImpl.java:564)
at java.base/java.net.Socket.connect(Socket.java:633)
at java.base/java.net.Socket.connect(Socket.java:583)
at java.base/sun.net.NetworkClient.doConnect(NetworkClient.java:183)
at java.base/sun.net.www.http.HttpClient.openServer(HttpClient.java:498)
at java.base/sun.net.www.http.HttpClient.openServer(HttpClient.java:603)
at java.base/sun.net.www.http.HttpClient.<init>(HttpClient.java:246)
at java.base/sun.net.www.http.HttpClient.New(HttpClient.java:351)
at java.base/sun.net.www.http.HttpClient.New(HttpClient.java:373)
at java.base/sun.net.www.protocol.http.HttpURLConnection.getNewHttpClient(HttpURLConnection.java:1309)
at java.base/sun.net.www.protocol.http.HttpURLConnection.plainConnect0(HttpURLConnection.java:1242)
at java.base/sun.net.www.protocol.http.HttpURLConnection.plainConnect(HttpURLConnection.java:1128)
at java.base/sun.net.www.protocol.http.HttpURLConnection.connect(HttpURLConnection.java:1057)
at java.base/sun.net.www.protocol.http.HttpURLConnection.getInputStream0(HttpURLConnection.java:1665)
at java.base/sun.net.www.protocol.http.HttpURLConnection.getInputStream(HttpURLConnection.java:1589)
at java.base/java.net.URL.openStream(URL.java:1161)
at org.apache.commons.io.FileUtils.copyURLToFile(FileUtils.java:1460)
at org.deeplearning4j.base.MnistFetcher.tryDownloadingAFewTimes(MnistFetcher.java:184)
at org.deeplearning4j.base.MnistFetcher.tryDownloadingAFewTimes(MnistFetcher.java:177)
at org.deeplearning4j.base.MnistFetcher.downloadAndUntar(MnistFetcher.java:156)
at org.deeplearning4j.datasets.fetchers.MnistDataFetcher.<init>(MnistDataFetcher.java:67)
at org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator.<init>(MnistDataSetIterator.java:67)
at org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator.<init>(MnistDataSetIterator.java:53)
at com.example.deeplearning4jcnn.Deeplearning4jCnnApplication.main(Deeplearning4jCnnApplication.java:39)
at java.base/jdk.internal.reflect.DirectMethodHandleAccessor.invoke(DirectMethodHandleAccessor.java:104)
... 2 more

Process finished with exit code 0

Mnist Fetcher ->

Hier Problem:
at org.deeplearning4j.base.MnistFetcher.tryDownloadingAFewTimes(MnistFetcher.java:184)
at org.deeplearning4j.base.MnistFetcher.tryDownloadingAFewTimes(MnistFetcher.java:177)
at org.deeplearning4j.base.MnistFetcher.downloadAndUntar(MnistFetcher.java:156)


    public File downloadAndUntar() throws IOException {
        if (this.fileDir != null) {
            return this.fileDir;
        } else {
            File baseDir = this.getBaseDir();
            if (!baseDir.isDirectory() && !baseDir.mkdir()) {
                throw new IOException("Could not mkdir " + baseDir);
            } else {
                log.info("Downloading {}...", this.getName());
                File tarFile = new File(baseDir, this.getTrainingFilesFilename());
                File testFileLabels = new File(baseDir, this.getTestFilesFilename());
                this.tryDownloadingAFewTimes(new URL(this.getTrainingFilesURL()), tarFile, this.getTrainingFilesMD5());
                this.tryDownloadingAFewTimes(new URL(this.getTestFilesURL()), testFileLabels, this.getTestFilesMD5());
                ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath(), baseDir.getAbsolutePath());
                ArchiveUtils.unzipFileTo(testFileLabels.getAbsolutePath(), baseDir.getAbsolutePath());
                File labels = new File(baseDir, this.getTrainingFileLabelsFilename());
                File labelsTest = new File(baseDir, this.getTestFileLabelsFilename());
                this.tryDownloadingAFewTimes(new URL(this.getTrainingFileLabelsURL()), labels, this.getTrainingFileLabelsMD5());
                this.tryDownloadingAFewTimes(new URL(this.getTestFileLabelsURL()), labelsTest, this.getTestFileLabelsMD5());
                ArchiveUtils.unzipFileTo(labels.getAbsolutePath(), baseDir.getAbsolutePath());
                ArchiveUtils.unzipFileTo(labelsTest.getAbsolutePath(), baseDir.getAbsolutePath());
                this.fileDir = baseDir;
                return this.fileDir;
            }
        }
    }

    -> this.tryDownloadingAFewTimes(new URL(this.getTrainingFilesURL()), tarFile, this.getTrainingFilesMD5());
    -> public String getTrainingFilesURL() { return "http://benchmark.deeplearn.online/mnist/train-images-idx3-ubyte.gz";}

ping benchmark.deeplearn.online --> nicht gefunden -> URL alt

--> Prolematisch wenn bei den Einstiegsbeispielen solche Fehler auftreten
--> Download Quelle sollte default haben aber änderbar sein
-->  EmnistDataSetIterator(EmnistDataSetIterator.Set.MNIST, batchSize, true); geht auch nicht 
Quelle Offizielle Quickstart Guide https://deeplearning4j.konduit.ai/v/en-1.0.0-beta7/getting-started/tutorials/quickstart-with-mnist



## Problem 2:

2022-11-28T18:11:05.717+01:00  INFO 28500 --- [  restartedMain] org.nd4j.linalg.factory.Nd4jBackend      : Loaded [CpuBackend] backend
2022-11-28T18:11:05.921+01:00  INFO 28500 --- [  restartedMain] org.nd4j.nativeblas.NativeOpsHolder      : Number of threads used for NativeOps: 8
A fatal error has been detected by the Java Runtime Environment:
EXCEPTION_ACCESS_VIOLATION (0xc0000005) at pc=0x0000000000000000, pid=28500, tid=5956
JRE version: Java(TM) SE Runtime Environment (18.0.2.1+1) (build 18.0.2.1+1-1)
Java VM: Java HotSpot(TM) 64-Bit Server VM (18.0.2.1+1-1, mixed mode, emulated-client, sharing, tiered, compressed oops, compressed class ptrs, g1 gc, windows-amd64)
Problematic frame:
C  0x0000000000000000
No core dump will be written. Minidumps are not enabled by default on client versions of Windows


--> Test auf Linux
java: cannot access org.springframework.boot.SpringApplication bad class file:
FIX:
System Java Version
==
pom <java.version>16</java.version>
==
pom maven compiler configuration java version    <source>16</source>  <target>16</target>
==
intellij idea -> File -> Project Structure -> Language Level and Module SDK same java version

--> Unter meinem Windows System Funktioniert diese Lösung nicht:
Windows 11
AMD Ryzen 7 4700U with Radeon Graphics

Anmerkung für vergleich -> Deep java library war nicht so penibel
-> egal wenn system java version anders als intellij project einstellungen
-> Und Windows und Linux haben beide Funktioniert


## Problem 3:

Cannot do forward pass in Convolution layer 
(layer name = layer0, layer index = 0): input array depth does not match CNN layer configuration (data input depth = 1, [minibatch,inputDepth,height,width]=[100, 1, 28, 28]; expected input depth = 3)

-> kurz: Layer 0 erwartet Tiefe 3 bekommt aber tiefe 1 

.nIn(3) ->  .nIn(1)

## Problem 4:

                .layer(2,
                        new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                                .activation(Activation.RELU)
                                .weightInit(WeightInit.XAVIER_UNIFORM)
                                .nOut(256)
                                .build())
                .layer(3,
                        new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                                .activation(Activation.SOFTMAX)
                                .weightInit(WeightInit.XAVIER_UNIFORM)
                                .nOut(10)
                                .build())

            FIX: -> Layer 2 von oben realisiter tnicht den Flatten Layer so wie ich mir das vorgestellt habe 
            .layer(2,
                        new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                                .activation(Activation.SOFTMAX)
                                .weightInit(WeightInit.XAVIER_UNIFORM)
                                .nOut(10)
                                .build())

