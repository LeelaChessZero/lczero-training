#!/usr/bin/env bash

protoc --proto_path=libs/lczero-common --python_out=tf libs/lczero-common/proto/net.proto
protoc --proto_path=libs/lczero-common --python_out=tf libs/lczero-common/proto/chunk.proto
touch tf/proto/__init__.py
echo "cd libs/lczero-common/proto && ls"
cd libs/lczero-common/proto && ls
echo proto files are successfully patched
