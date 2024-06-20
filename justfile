#!/bin/bash

run-test TEST:
	cargo test --test {{TEST}}

build:
	cargo build --all

run:
	cargo run --profile dev --bin zkml

tests:
	cargo test --all

bench:
	cargo bench

format:
	cargo fmt

lint:
	cargo clippy
