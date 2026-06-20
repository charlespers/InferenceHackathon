use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("model forward pass failed: {0}")]
    Forward(String),

    #[error("speculative decoding error: {0}")]
    Spec(String),

    #[error("CUDA error: {0}")]
    Cuda(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, EngineError>;
