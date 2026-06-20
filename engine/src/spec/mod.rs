pub mod accept;
pub mod adaptive_verify;
pub mod draft_vocab;
pub mod eagle3_engine;
pub mod engine;
pub mod model;
pub mod route_aware;
pub mod route_aware_drafter;
pub mod types;

pub use adaptive_verify::{adaptive_verify_depth, emitted, expected_accepted, verify_cost, VerifyPlan};
pub use draft_vocab::DraftVocabMap;
pub use eagle3_engine::{AuxModelRunner, Eagle3Config, Eagle3Engine, Eagle3RoundStats};
pub use engine::{SpecConfig, SpecEngine, RoundStats};
pub use model::{DrafterPool, ModelRunner};
pub use route_aware::{Candidate, ExpertUnion, RouteAwarePolicy};
pub use route_aware_drafter::{CandidateSource, RouteAwareDrafter};
pub use types::{AcceptedRun, DraftProposal, DraftTree, RngCore, TargetLogits, TokenId};
