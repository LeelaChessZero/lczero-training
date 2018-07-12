#!/usr/bin/env bash

pushd libs/lczero-common
protoc --python_out=../../tf lc0net.proto
popd
