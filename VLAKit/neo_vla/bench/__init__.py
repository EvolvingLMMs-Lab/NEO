"""Simulation benchmarks for released and trained steps.

Each benchmark runs its simulator in its own environment and asks a running
neo-vla-serve for actions over HTTP, so nothing here imports torch. What a
benchmark needs to know about the step it scores (cameras, action frames,
unit factors, protocol settings) is read from the step's eval.json.
"""
