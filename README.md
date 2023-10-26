# how-2-cli

A rough idea of how to replicate Copilot's CLI mode using Code LLaMA2

Didn't quite pan out, for the following reasons;

- Rustformer's llm outputs a bunch of GGML output.
- Hard to nail down only outputting commands without system prompts support
- Long startup time, but could be mitigated by snapshotting the prompt.

Overall, the startup time made it less useful overall. Perhaps a fine-tuned model for bash specifically could work better.

## Next Steps?

- Try another LLaMA 2 impl? Burn? Candle? So many choices.

For now, I'll just have to memorise commands, using my meat brain. 
