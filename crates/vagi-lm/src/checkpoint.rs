//! Checkpoint save/load for model weights and optimizer state.
//!
//! File format:
//! ```text
//! [4 bytes: header_len as u32 LE]
//! [header_len bytes: JSON header]
//! [N * 4 bytes: f32 model weights]
//! [N * 4 bytes: f32 adam_m (optional)]
//! [N * 4 bytes: f32 adam_v (optional)]
//! ```

use std::io::Write;
use crate::config::LMConfig;
use crate::model::VagiLM;
use crate::training::{LMTrainer, AdvancedConfig};

/// Checkpoint header (serialized as JSON).
struct CheckpointHeader {
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    ffn_dim: usize,
    vocab_size: usize,
    max_seq_len: usize,
    n_weights: usize,
    has_optimizer: bool,
    step: usize,
    lr: f32,
}

impl CheckpointHeader {
    fn to_json(&self) -> String {
        format!(
            r#"{{"d_model":{},"n_layers":{},"n_heads":{},"ffn_dim":{},"vocab_size":{},"max_seq_len":{},"n_weights":{},"has_optimizer":{},"step":{},"lr":{}}}"#,
            self.d_model, self.n_layers, self.n_heads, self.ffn_dim,
            self.vocab_size, self.max_seq_len, self.n_weights,
            self.has_optimizer, self.step, self.lr,
        )
    }

    fn from_json(s: &str) -> Option<Self> {
        // Minimal JSON parsing without serde
        let get = |key: &str| -> Option<&str> {
            let pat = format!("\"{}\":", key);
            let start = s.find(&pat)? + pat.len();
            let rest = &s[start..];
            let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
            Some(rest[..end].trim())
        };
        Some(Self {
            d_model: get("d_model")?.parse().ok()?,
            n_layers: get("n_layers")?.parse().ok()?,
            n_heads: get("n_heads")?.parse().ok()?,
            ffn_dim: get("ffn_dim")?.parse().ok()?,
            vocab_size: get("vocab_size")?.parse().ok()?,
            max_seq_len: get("max_seq_len")?.parse().ok()?,
            n_weights: get("n_weights")?.parse().ok()?,
            has_optimizer: get("has_optimizer")? == "true",
            step: get("step")?.parse().ok()?,
            lr: get("lr")?.parse().ok()?,
        })
    }
}

/// Collect all model weights into a flat f32 vector.
fn collect_weights(model: &VagiLM) -> Vec<f32> {
    let mut w = Vec::new();
    w.extend_from_slice(&model.embedding.weight);
    for layer in &model.layers {
        w.extend_from_slice(&layer.attention.wq.w_latent);
        w.extend_from_slice(&layer.attention.wk.w_latent);
        w.extend_from_slice(&layer.attention.wv.w_latent);
        w.extend_from_slice(&layer.attention.wo.w_latent);
        w.extend_from_slice(&layer.ffn_up.w_latent);
        w.extend_from_slice(&layer.ffn_down.w_latent);
    }
    w.extend_from_slice(&model.lm_head.w_latent);
    w
}

/// Restore weights from a flat f32 vector back into the model.
fn restore_weights(model: &mut VagiLM, w: &[f32]) {
    let mut offset = 0;

    let n = model.embedding.weight.len();
    model.embedding.weight.copy_from_slice(&w[offset..offset + n]);
    offset += n;

    for layer in &mut model.layers {
        let n = layer.attention.wq.w_latent.len();
        layer.attention.wq.w_latent.copy_from_slice(&w[offset..offset + n]);
        offset += n;
        let n = layer.attention.wk.w_latent.len();
        layer.attention.wk.w_latent.copy_from_slice(&w[offset..offset + n]);
        offset += n;
        let n = layer.attention.wv.w_latent.len();
        layer.attention.wv.w_latent.copy_from_slice(&w[offset..offset + n]);
        offset += n;
        let n = layer.attention.wo.w_latent.len();
        layer.attention.wo.w_latent.copy_from_slice(&w[offset..offset + n]);
        offset += n;
        let n = layer.ffn_up.w_latent.len();
        layer.ffn_up.w_latent.copy_from_slice(&w[offset..offset + n]);
        offset += n;
        let n = layer.ffn_down.w_latent.len();
        layer.ffn_down.w_latent.copy_from_slice(&w[offset..offset + n]);
        offset += n;
    }

    let n = model.lm_head.w_latent.len();
    model.lm_head.w_latent.copy_from_slice(&w[offset..offset + n]);
}

fn f32_to_bytes(data: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    }
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    assert!(data.len() % 4 == 0);
    let n = data.len() / 4;
    let mut out = vec![0.0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), out.as_mut_ptr() as *mut u8, data.len());
    }
    out
}

/// Save model weights only (no optimizer state). Lightweight.
pub fn save_model(model: &VagiLM, path: &str) -> std::io::Result<usize> {
    let weights = collect_weights(model);
    let header = CheckpointHeader {
        d_model: model.config.d_model,
        n_layers: model.config.n_layers,
        n_heads: model.config.n_heads,
        ffn_dim: model.config.ffn_dim,
        vocab_size: model.config.vocab_size,
        max_seq_len: model.config.max_seq_len,
        n_weights: weights.len(),
        has_optimizer: false,
        step: 0,
        lr: 0.0,
    };
    write_checkpoint(path, &header, &weights, None, None)
}

/// Save full checkpoint: model + optimizer state for resume.
pub fn save_checkpoint(
    model: &VagiLM,
    trainer: &LMTrainer,
    path: &str,
) -> std::io::Result<usize> {
    let weights = collect_weights(model);
    let header = CheckpointHeader {
        d_model: model.config.d_model,
        n_layers: model.config.n_layers,
        n_heads: model.config.n_heads,
        ffn_dim: model.config.ffn_dim,
        vocab_size: model.config.vocab_size,
        max_seq_len: model.config.max_seq_len,
        n_weights: weights.len(),
        has_optimizer: true,
        step: trainer.step_count(),
        lr: trainer.current_lr(),
    };
    write_checkpoint(path, &header, &weights, Some(&trainer.adam_m), Some(&trainer.adam_v))
}

fn write_checkpoint(
    path: &str,
    header: &CheckpointHeader,
    weights: &[f32],
    adam_m: Option<&[f32]>,
    adam_v: Option<&[f32]>,
) -> std::io::Result<usize> {
    let json = header.to_json();
    let json_bytes = json.as_bytes();
    let header_len = json_bytes.len() as u32;

    let mut file = std::fs::File::create(path)?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(json_bytes)?;
    file.write_all(f32_to_bytes(weights))?;

    let mut total = 4 + json_bytes.len() + weights.len() * 4;
    if let (Some(m), Some(v)) = (adam_m, adam_v) {
        file.write_all(f32_to_bytes(m))?;
        file.write_all(f32_to_bytes(v))?;
        total += m.len() * 4 + v.len() * 4;
    }
    Ok(total)
}

/// Load model weights from checkpoint (inference only).
pub fn load_model(path: &str) -> std::io::Result<VagiLM> {
    let (header, weights, _, _) = read_checkpoint(path)?;
    let config = LMConfig {
        d_model: header.d_model,
        n_layers: header.n_layers,
        n_heads: header.n_heads,
        ffn_dim: header.ffn_dim,
        vocab_size: header.vocab_size,
        max_seq_len: header.max_seq_len,
        rms_eps: 1e-6,
    };
    let mut model = VagiLM::new(config);
    if weights.len() == header.n_weights {
        restore_weights(&mut model, &weights);
    }
    Ok(model)
}

/// Load full checkpoint for resuming training.
pub fn load_checkpoint(
    path: &str,
    train_cfg: AdvancedConfig,
) -> std::io::Result<(VagiLM, LMTrainer)> {
    let (header, weights, adam_m, adam_v) = read_checkpoint(path)?;
    let config = LMConfig {
        d_model: header.d_model,
        n_layers: header.n_layers,
        n_heads: header.n_heads,
        ffn_dim: header.ffn_dim,
        vocab_size: header.vocab_size,
        max_seq_len: header.max_seq_len,
        rms_eps: 1e-6,
    };
    let mut model = VagiLM::new(config);
    if weights.len() == header.n_weights {
        restore_weights(&mut model, &weights);
    }

    let mut trainer = LMTrainer::new(&model, train_cfg);
    if let (Some(m), Some(v)) = (adam_m, adam_v) {
        if m.len() == trainer.adam_m.len() {
            trainer.adam_m.copy_from_slice(&m);
            trainer.adam_v.copy_from_slice(&v);
            trainer.step = header.step;
        }
    }
    Ok((model, trainer))
}

fn read_checkpoint(
    path: &str,
) -> std::io::Result<(CheckpointHeader, Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>)> {
    let data = std::fs::read(path)?;
    if data.len() < 4 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "File too small"));
    }

    let header_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    if data.len() < 4 + header_len {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Truncated header"));
    }

    let json_str = std::str::from_utf8(&data[4..4 + header_len])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let header = CheckpointHeader::from_json(json_str)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Bad JSON header"))?;

    let weights_start = 4 + header_len;
    let weights_bytes = header.n_weights * 4;
    if data.len() < weights_start + weights_bytes {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Truncated weights"));
    }
    let weights = bytes_to_f32(&data[weights_start..weights_start + weights_bytes]);

    let (adam_m, adam_v) = if header.has_optimizer {
        let m_start = weights_start + weights_bytes;
        let m_end = m_start + weights_bytes;
        let v_end = m_end + weights_bytes;
        if data.len() >= v_end {
            (
                Some(bytes_to_f32(&data[m_start..m_end])),
                Some(bytes_to_f32(&data[m_end..v_end])),
            )
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    Ok((header, weights, adam_m, adam_v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_model() {
        let config = LMConfig::tiny();
        let model = VagiLM::new(config);
        let path = std::env::temp_dir().join("vagi_test_model.ckpt");
        let path = path.to_str().unwrap();
        save_model(&model, path).unwrap();
        let loaded = load_model(path).unwrap();
        assert_eq!(loaded.config.d_model, model.config.d_model);
        assert_eq!(loaded.config.n_layers, model.config.n_layers);
        // Check weights match
        let w1 = collect_weights(&model);
        let w2 = collect_weights(&loaded);
        assert_eq!(w1.len(), w2.len());
        for (a, b) in w1.iter().zip(w2.iter()) {
            assert!((a - b).abs() < 1e-10, "Weight mismatch: {a} vs {b}");
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_load_checkpoint() {
        let config = LMConfig::tiny();
        let model = VagiLM::new(config);
        let adv = AdvancedConfig::default();
        let mut trainer = LMTrainer::new(&model, adv.clone());
        // Simulate some steps
        trainer.step = 42;
        trainer.adam_m[0] = 1.23;
        trainer.adam_v[0] = 4.56;

        let path = std::env::temp_dir().join("vagi_test_ckpt.ckpt");
        let path = path.to_str().unwrap();
        save_checkpoint(&model, &trainer, path).unwrap();

        let (loaded_model, loaded_trainer) = load_checkpoint(path, adv).unwrap();
        assert_eq!(loaded_trainer.step_count(), 42);
        assert!((loaded_trainer.adam_m[0] - 1.23).abs() < 1e-6);
        assert!((loaded_trainer.adam_v[0] - 4.56).abs() < 1e-6);

        let w1 = collect_weights(&model);
        let w2 = collect_weights(&loaded_model);
        for (a, b) in w1.iter().zip(w2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_header_json_roundtrip() {
        let h = CheckpointHeader {
            d_model: 64, n_layers: 4, n_heads: 4, ffn_dim: 256,
            vocab_size: 259, max_seq_len: 256, n_weights: 12345,
            has_optimizer: true, step: 100, lr: 0.005,
        };
        let json = h.to_json();
        let h2 = CheckpointHeader::from_json(&json).unwrap();
        assert_eq!(h2.d_model, 64);
        assert_eq!(h2.n_weights, 12345);
        assert_eq!(h2.step, 100);
        assert!(h2.has_optimizer);
    }
}
