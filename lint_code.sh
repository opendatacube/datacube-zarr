#!/usr/bin/env bash
# Convenience script for linting code before committing.

isort -rc . && black --skip-string-normalization --line-length 90 .