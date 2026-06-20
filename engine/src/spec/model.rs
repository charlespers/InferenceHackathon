/// The single interface the speculative engine requires of a model.
///
/// Conifer (or any other backend) implements this for both the draft models
/// and the target model. The engine calls nothing else.
///
/// # GPU note
/// The implementor owns its CUDA context and weight buffers. `forward` is
/// responsible for keeping activations on-device; it copies logits to host
/// only for the positions the caller requests (controlled by `positions`).
/// For draft models at B=1 that means one logit row per step.
/// For the target verifier it means N*k rows in one batched call.
pub trait ModelRunner: Send + Sync {
    /// Run a single auto-regressive decode step.
    ///
    /// `context` : token IDs seen so far (the KV cache is assumed warm).
    /// `next_token` : the one new token being decoded.
    /// Returns logits over the full vocabulary for `next_token`'s position.
    fn forward_single(&self, context: &[u32], next_token: u32)
        -> crate::error::Result<Vec<f32>>;

    /// Batched verification forward pass for speculative decoding.
    ///
    /// `context` : the confirmed prefix (KV cache is warm up to here).
    /// `draft_tokens` : flat slice of N*k draft tokens (tree, row-major by drafter).
    /// `vocab_size` : vocabulary size (for sizing the output).
    ///
    /// Returns logits shaped [N*k, vocab_size], row-major. The caller
    /// (SpecEngine) is responsible for the acceptance logic.
    ///
    /// Default impl: calls `forward_single` N*k times sequentially.
    /// Override with a true batched CUDA kernel for real performance.
    fn forward_batch(
        &self,
        context: &[u32],
        draft_tokens: &[u32],
        vocab_size: usize,
    ) -> crate::error::Result<crate::spec::types::TargetLogits> {
        let n_positions = draft_tokens.len();
        let mut data = Vec::with_capacity(n_positions * vocab_size);
        let mut ctx: Vec<u32> = context.to_vec();
        for &tok in draft_tokens {
            let logits = self.forward_single(&ctx, tok)?;
            assert_eq!(logits.len(), vocab_size, "vocab size mismatch");
            data.extend_from_slice(&logits);
            ctx.push(tok);
        }
        Ok(crate::spec::types::TargetLogits { data, n_positions, vocab_size })
    }

    fn vocab_size(&self) -> usize;
}

/// A pool of N draft model runners that can execute in parallel.
///
/// On the 8×H100 node each drafter lives on its own GPU (or shares one with
/// other small drafters). The pool schedules them concurrently via tokio tasks.
/// With FP8 target weights (235 GB) there is ~405 GB of HBM headroom —
/// enough for many copies of a Qwen3-1.7B drafter (~3.4 GB each).
pub trait DrafterPool: Send + Sync {
    /// Draft `draft_len` tokens from each of the N drafters in parallel,
    /// given the current confirmed context.
    ///
    /// Returns one `DraftProposal` per drafter.
    fn draft(
        &self,
        context: &[u32],
        draft_len: usize,
    ) -> crate::error::Result<Vec<crate::spec::types::DraftProposal>>;

    fn n_drafters(&self) -> usize;
}
