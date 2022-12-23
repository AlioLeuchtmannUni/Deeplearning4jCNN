#!/bin/bash

echo 'starting spring boot'

cd code

echo 'installing dependencies'

./mvnw clean install -X

echo 'start programm'

./mvnw spring-boot:run -X


