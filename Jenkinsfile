pipeline {
    agent any

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                python -m pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Unit Tests') {
            steps {
                sh 'pytest -v'
            }
        }

        stage('Code Quality') {
            when {
                branch 'main'
            }
            steps {
                sh 'flake8 .'
            }
        }
    }

    post {
        success {
            echo "CI passed on branch: ${env.BRANCH_NAME}"
        }
        failure {
            echo "CI failed on branch: ${env.BRANCH_NAME}"
        }
    }
}
