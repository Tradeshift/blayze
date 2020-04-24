pipeline {
    agent any
    triggers {
        issueCommentTrigger('^retest$')
    }

    tools {
       jdk 'openjdk8-jdk'
       maven 'apache-maven-3.5.0'
    }
    environment {
        P12_PASSWORD = credentials 'client-cert-password'
        MAVEN_OPTS = "-Djavax.net.ssl.keyStore=/var/lib/jenkins/.m2/certs/jenkins.p12 \
                      -Djavax.net.ssl.keyStoreType=pkcs12 \
                      -Djavax.net.ssl.keyStorePassword=$P12_PASSWORD"
    }

    stages {
        stage('Clone') {
            steps {
                checkout scm
            }
        }
        stage('Build') {
            steps {
                sh 'mvn dependency:go-offline'
                sh 'mvn clean verify'
            }
        }
    }
}
