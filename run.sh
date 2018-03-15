#!/bin/bash


mvn install
mvn org.apache.maven.plugins:maven-install-plugin:2.5.2:install-file -Dfile=target/DQN-1.0-SNAPSHOT.jar -DpomFile=pom.xml
