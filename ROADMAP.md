# ForgeOS Roadmap

## Post-Self-Hosting Goal: AI-in-the-Loop

Once Forge can parse and evaluate itself, the next milestone is **Claude running inside ForgeOS**:

1. Connect via OAuth (borrow token from claude-code or spacebot)
2. Claude operates through the ForgeOS HTTP API — emitting mutations, running evals, querying program state
3. Claude modifies the Forge runtime FROM INSIDE Forge — writing parser improvements in Forge that the Forge parser then uses
4. A self-modifying AI in its own environment

The loop: Claude writes Forge code → ForgeOS validates and runs it → results feed back to Claude → Claude improves the code → the improved code improves ForgeOS itself.

This is the design doc's Phase 8 vision made concrete.
