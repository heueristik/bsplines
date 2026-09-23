# List the recipes.
default:
    just --list

# Build all targets.
build:
    cargo build --all-targets --locked

# Run all tests.
test:
    cargo test --locked

# Lint all targets. Warnings are errors.
clippy:
    cargo clippy --all-targets --locked -- -D warnings

# Build the documentation. Warnings are errors.
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --package bsplines --features doc-images --no-deps --locked

# Format the Rust code.
fmt:
    cargo +nightly fmt --all

# Check the Rust formatting.
fmt-check:
    cargo +nightly fmt --all -- --check

# Format the TOML files.
taplo:
    taplo fmt

# Check the TOML formatting.
taplo-check:
    taplo fmt --check --diff

# Run all checks that CI runs.
ci: build test clippy doc fmt-check taplo-check
