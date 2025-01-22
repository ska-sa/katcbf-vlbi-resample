// Copyright (c) 2025, National Research Foundation (SARAO)

pipeline {
  // TODO: the agents should probably use a more generic label like 'gpu'
  agent {
    dockerfile {
      registryCredentialsId 'dockerhub'  // Supply credentials to avoid rate limit
      filename 'Dockerfile.jenkins'
      args '--runtime=nvidia --gpus=all'
      label 'katgpucbf'
    }
  }

  options {
    timeout(time: 1, unit: 'HOURS')
    disableConcurrentBuilds(abortPrevious: true)
  }

  stages {
    stage('Create virtual environment') {
      steps {
        sh "python3 -m venv /tmp/venv"
      }
    }

    stage('Install') {
      steps {
        // Use a binary wheel instead of compiling from source, to
        // speed things up
        sh "sed 's/^cupy==/cupy-cuda12x==/' requirements.txt > requirements-ci.txt"
        sh "pip install -r requirements-ci.txt"
        sh "pip install --no-deps -e ."
      }
    }

    stage('Parallel stage') {
      parallel {

        stage('Run pre-commit checks') {
          steps {
            sh 'SKIP=no-commit-to-branch pre-commit run --all-files'
          }
        }

        stage('Test') {
          steps {
            sh "pytest -v -ra --junitxml=result.xml --cov=test --cov=katcbf_vlbi_resample --cov-report=xml --cov-branch --suppress-tests-failed-exit-code"
            junit 'result.xml'
            cobertura coberturaReportFile: 'coverage.xml'
          }
        }

        stage('Build documentation') {
          steps {
            // -W causes warnings to become errors.
            // --keep-going ensures we get all warnings instead of just the first.
            sh 'make -C doc clean html SPHINXOPTS="-W --keep-going"'
              publishHTML(target: [reportName: 'Documentation', reportDir: 'doc/_build/html', reportFiles: 'index.html'])
          }
        }

      }
    }   // end of parallel stage

  }

  post {
    always {
      emailext attachLog: true,
      attachmentsPattern: 'reports/result.xml',
      body: '${SCRIPT, template="groovy-html.template"}',
      recipientProviders: [developers(), requestor(), culprits()],
      subject: '$PROJECT_NAME - $BUILD_STATUS!',
      to: '$DEFAULT_RECIPIENTS'

      cleanWs()
    }
  }
}
