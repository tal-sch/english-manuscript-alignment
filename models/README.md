# Selected model checkpoint

`manuscript-registration-best.pt` is the selected checkpoint used by the command-line
inference and Gradio application.

- Architecture: `PatchCorrelationRegistration`
- Training: 30 mixed IAM/cross-font epochs followed by 8 identity fine-tuning epochs
- Input training canvas: `96 x 512`
- File size: 13,463,477 bytes
- SHA-256: `44AC5477F37F3BB3C4218C84270A3014E54B4B8CEF0B6D0924AEC3B6CFC58E2A`

Intermediate checkpoints and optimizer states remain excluded from Git under
`registration_runs/`.
