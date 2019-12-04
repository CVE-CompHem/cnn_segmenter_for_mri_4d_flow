#!/bin/bash

set -eoux pipefail

DOWNLOAD_DIR=../tmp
DATASOURCE_DIR=../data
DATASOURCE_FILENAME=126_subjects.zip
SEGMENTATION_FILENAME=freiburg_subjects_with_nicolas_rw_segmentations.zip

if [ ! -d $DATASOURCE_DIR ]; then
  echo "Downloading files and extracting them to $(pwd)/${DATASOURCE_DIR} directory." 
  mkdir -p ${DOWNLOAD_DIR}
  mkdir -p ${DATASOURCE_DIR}
  wget -O ${DOWNLOAD_DIR}/${DATASOURCE_FILENAME} https://polybox.ethz.ch/index.php/s/OdNVbDw41F0hHny/download 
  unzip ${DOWNLOAD_DIR}/${DATASOURCE_FILENAME} -d ${DATASOURCE_DIR}
  wget -O ${DOWNLOAD_DIR}/${SEGMENTATION_FILENAME} https://polybox.ethz.ch/index.php/s/HpLy7Re5mfXmMiP/download
  unzip ${DOWNLOAD_DIR}/${SEGMENTATION_FILENAME} -d ${DATASOURCE_DIR}
  #rm -r  ${DOWNLOAD_DIR}
else
  echo "${DATASOURCE_DIR} directory already exists - no files downloaded."
fi
