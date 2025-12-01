#!/usr/bin/env bash

protoc --proto_path=libs/lc0 --python_out=tf proto/net.proto
touch tf/proto/__init__.py
