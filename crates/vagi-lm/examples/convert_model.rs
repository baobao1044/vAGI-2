//! Convert raw weight file to checkpoint format.
//! Run once: cargo run --example convert_model -p vagi-lm --release

use std::io::Write;
use vagi_lm::{VagiLM, LMConfig};

fn main() {
    let raw_path = "data/vi_model.bin";
    let out_path = "data/vi_model_ckpt.bin";

    println!("Converting raw weights to checkpoint format...");

    let config = LMConfig::tiny();
    let mut model = VagiLM::new(config);

    // Load raw f32 weights
    let data = std::fs::read(raw_path).expect("Cannot read raw model");
    let n_f32 = data.len() / 4;
    let weights: Vec<f32> = (0..n_f32).map(|i| {
        let b = [data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]];
        f32::from_le_bytes(b)
    }).collect();

    println!("  Raw: {} bytes, {} f32 values", data.len(), weights.len());

    // Restore weights into model
    let mut offset = 0;
    let n = model.embedding.weight.len();
    model.embedding.weight.copy_from_slice(&weights[offset..offset+n]);
    offset += n;
    for layer in &mut model.layers {
        let n = layer.attention.wq.w_latent.len();
        layer.attention.wq.w_latent.copy_from_slice(&weights[offset..offset+n]); offset += n;
        let n = layer.attention.wk.w_latent.len();
        layer.attention.wk.w_latent.copy_from_slice(&weights[offset..offset+n]); offset += n;
        let n = layer.attention.wv.w_latent.len();
        layer.attention.wv.w_latent.copy_from_slice(&weights[offset..offset+n]); offset += n;
        let n = layer.attention.wo.w_latent.len();
        layer.attention.wo.w_latent.copy_from_slice(&weights[offset..offset+n]); offset += n;
        let n = layer.ffn_up.w_latent.len();
        layer.ffn_up.w_latent.copy_from_slice(&weights[offset..offset+n]); offset += n;
        let n = layer.ffn_down.w_latent.len();
        layer.ffn_down.w_latent.copy_from_slice(&weights[offset..offset+n]); offset += n;
    }
    let n = model.lm_head.w_latent.len();
    model.lm_head.w_latent.copy_from_slice(&weights[offset..offset+n]);
    offset += n;
    println!("  Loaded {} / {} weights", offset, weights.len());

    // Save with checkpoint format
    match vagi_lm::save_model(&model, out_path) {
        Ok(sz) => println!("  Saved {out_path} ({:.1} KB)", sz as f64 / 1024.0),
        Err(e) => eprintln!("  Error: {e}"),
    }

    // Also overwrite original
    match vagi_lm::save_model(&model, raw_path) {
        Ok(sz) => println!("  Overwrote {raw_path} ({:.1} KB)", sz as f64 / 1024.0),
        Err(e) => eprintln!("  Error: {e}"),
    }

    // Verify by loading
    match vagi_lm::load_model(raw_path) {
        Ok(m) => println!("  Verify: loaded OK (d={}, L={})", m.config.d_model, m.config.n_layers),
        Err(e) => eprintln!("  Verify FAIL: {e}"),
    }

    println!("Done!");
    std::io::stdout().flush().ok();
}
