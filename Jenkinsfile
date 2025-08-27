/* Copyright (c) 2025, National Research Foundation (SARAO)
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use
 * this file except in compliance with the License. You may obtain a copy
 * of the License at
 *
 *   https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
        sh "pip install -r requirements.txt"
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
            recordCoverage sourceCodeEncoding: 'UTF-8', tools: [[parser: 'COBERTURA', pattern: 'coverage.xml']]
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
