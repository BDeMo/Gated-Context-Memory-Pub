# Checkpoint availability

The intended upstream location is
https://huggingface.co/BDeM/Gated-Context-Memory/tree/main/checkpoints .
An unauthenticated access check during release preparation failed. Public access,
file hashes, backbone revisions, and mappings from checkpoints to paper results
are not yet verified. No weights are redistributed by this repository.

You can train an adapter using the README tutorial. Before evaluating an existing
adapter, verify its provenance and match its backbone, writer depth, state budget,
projection, and reader-adapter configuration. Do not substitute one run's checkpoint
or evaluation protocol for another when reproducing a table.

A completed release should provide a manifest with checkpoint path, SHA-256,
base-model revision, training configuration, evaluation command, and table/figure
identifier for each published result.
