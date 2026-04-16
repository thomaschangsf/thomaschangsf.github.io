#!/bin/bash
BASE_URL="https://raw.githubusercontent.com/thomaschangsf/thomaschangsf.github.io/master/site-professional/content/books"
SITES_PATH="sites/thomas_chang/content/books"

# Machine Learning
mkdir -p "$SITES_PATH/machine-learning"
curl -L "$BASE_URL/machine-learning/_index.md" -o "$SITES_PATH/machine-learning/_index.md"
curl -L "$BASE_URL/machine-learning/data-science-from-scratch.md" -o "$SITES_PATH/machine-learning/data-science-from-scratch.md"
curl -L "$BASE_URL/machine-learning/essential-math-for-data-science.md" -o "$SITES_PATH/machine-learning/essential-math-for-data-science.md"
curl -L "$BASE_URL/machine-learning/hands-on-large-language-model.md" -o "$SITES_PATH/machine-learning/hands-on-large-language-model.md"
curl -L "$BASE_URL/machine-learning/machine-learning-system-design-interview.md" -o "$SITES_PATH/machine-learning/machine-learning-system-design-interview.md"

# Software Profession
mkdir -p "$SITES_PATH/software-profession"
curl -L "$BASE_URL/software-profession/_index.md" -o "$SITES_PATH/software-profession/_index.md"
curl -L "$BASE_URL/software-profession/software-engineering-at-google.md" -o "$SITES_PATH/software-profession/software-engineering-at-google.md"
curl -L "$BASE_URL/software-profession/software-mistakes-and-tradeoffs.md" -o "$SITES_PATH/software-profession/software-mistakes-and-tradeoffs.md"
curl -L "$BASE_URL/software-profession/staff-engineers-path.md" -o "$SITES_PATH/software-profession/staff-engineers-path.md"

# Software System
mkdir -p "$SITES_PATH/software-system"
curl -L "$BASE_URL/software-system/_index.md" -o "$SITES_PATH/software-system/_index.md"
curl -L "$BASE_URL/software-system/getting-started-with-bazel.md" -o "$SITES_PATH/software-system/getting-started-with-bazel.md"
curl -L "$BASE_URL/software-system/grpc-up-and-running.md" -o "$SITES_PATH/software-system/grpc-up-and-running.md"
curl -L "$BASE_URL/software-system/kubernetes-up-and-running.md" -o "$SITES_PATH/software-system/kubernetes-up-and-running.md"
curl -L "$BASE_URL/software-system/learning-algorithms.md" -o "$SITES_PATH/software-system/learning-algorithms.md"
curl -L "$BASE_URL/software-system/learning-go.md" -o "$SITES_PATH/software-system/learning-go.md"
curl -L "$BASE_URL/software-system/scaling-ml-spark.md" -o "$SITES_PATH/software-system/scaling-ml-spark.md"

echo "Books content migrated successfully."
