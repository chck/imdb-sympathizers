CLOUDSDK_PYTHON:=$(HOME)/.anyenv/envs/pyenv/shims/python
GCLOUD_PROJECT:=YOUR_PROJECT_ID
_:=$(shell gcloud config set project $(GCLOUD_PROJECT))
TIMESTAMP:=$(shell date +"%Y%m%d%H%M%S")
JOB_NAME:=cloudml_imdb_$(TIMESTAMP)
GCS_PATH:=gs://YOUR_GCS_BUCKET/AND_PATH
REGION:=us-east1
BATCH_SIZE:=128
EPOCHS:=4

.PHONY: all
all: help

.PHONY: train-local ## Train on Local GPU
train-local:
	PYTHONPATH=. nohup python3 trainer/task.py \
	--model_name $(MODEL) \
	--gcs_path $(GCS_PATH) \
	--optimizer $(OPT) &

.PHONY: train-repl ## Train on Local GPU
train-repl:
	PYTHONPATH=. python3 trainer/task.py \
	--model_name $(MODEL) \
	--gcs_path $(GCS_PATH) \
	--optimizer $(OPT)

.PHONY: auth-tpu ## Authorize Cloud TPU
auth-tpu:
	$(eval TPU_ACCOUNT := $(shell curl -H "Authorization: Bearer `gcloud auth print-access-token`" https://ml.googleapis.com/v1/projects/${GCLOUD_PROJECT}:getConfig | jq ".config.tpuServiceAccount"))
	gcloud projects add-iam-policy-binding $(GCLOUD_PROJECT) --member serviceAccount:$(TPU_ACCOUNT) --role roles/ml.serviceAgent

.PHONY: ml-local ## Train by CloudML on Local
ml-local:
	CLOUDSDK_PYTHON=$(CLOUDSDK_PYTHON) gcloud ml-engine local train \
	--distributed \
	--module-name trainer.task \
	--package-path trainer -- \
	--batch_size $(BATCH_SIZE) \
	--gcs_path $(GCS_PATH) \
	--epochs $(EPOCHS)

.PHONY: ml-remote ## Train by CloudML on Remote GPU
ml-remote:
	gcloud ml-engine jobs submit training $(JOB_NAME) \
	--module-name trainer.task \
	--package-path trainer \
	--config trainer/config.yml \
	--staging-bucket gs://ailab-ephemeral \
	--region $(REGION) -- \
	--batch_size $(BATCH_SIZE) \
	--gcs_path $(GCS_PATH) \
	--epochs $(EPOCHS)

.PHONY: jobs ## Show trainer by CloudML
jobs:
	gcloud ml-engine jobs list

.PHONY: test ## Run tests by pytest
test:
	py.test tests/

.PHONY: help ## View help
help:
	@grep -E '^.PHONY: [a-zA-Z_-]+.*?## .*$$' $(MAKEFILE_LIST) | sed 's/^.PHONY: //g' | awk 'BEGIN {FS = "## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
